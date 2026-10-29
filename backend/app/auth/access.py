"""
Project-level authorization.

Everything is scoped to the user's workspace first: projects in another
workspace do not exist as far as the user can tell. Inside their workspace,
admins can do everything. Clients can read a project when it is visible to the
whole workspace or when they are a member; writing (uploading, deleting,
re-chunking) requires admin or an "editor" membership.
"""

from typing import List, Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.rag import Project, ProjectMember
from app.models.user import User, UserRole


def same_workspace(user: User, project: Project) -> bool:
    return user.workspace_id is not None and project.workspace_id == user.workspace_id


def _membership(session: Session, project_id: int, user_id: int) -> Optional[ProjectMember]:
    return session.exec(
        select(ProjectMember)
        .where(ProjectMember.project_id == project_id)
        .where(ProjectMember.user_id == user_id)
    ).first()


def can_read_project(session: Session, user: User, project: Project) -> bool:
    if not same_workspace(user, project):
        return False
    if user.role == UserRole.ADMIN:
        return True
    if (project.visibility or "all_users") == "all_users":
        return True
    return _membership(session, project.id, user.id) is not None


def can_write_project(session: Session, user: User, project: Project) -> bool:
    if not same_workspace(user, project):
        return False
    if user.role == UserRole.ADMIN:
        return True
    member = _membership(session, project.id, user.id)
    return member is not None and member.role == "editor"


def require_project(session: Session, user: User, project_id: Optional[int], write: bool = False) -> Project:
    """Load a project the user may access, or raise 404/403.
    Unknown, foreign-workspace and forbidden projects all return 404 for readers
    so project IDs cannot be enumerated."""
    if project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    project = session.get(Project, project_id)
    if not project or not can_read_project(session, user, project):
        raise HTTPException(status_code=404, detail="Project not found")
    if write and not can_write_project(session, user, project):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this project")
    return project


def readable_project_ids(session: Session, user: User) -> List[int]:
    if user.workspace_id is None:
        return []
    in_workspace = select(Project.id).where(Project.workspace_id == user.workspace_id)
    if user.role == UserRole.ADMIN:
        return list(session.exec(in_workspace).all())
    public = session.exec(in_workspace.where(Project.visibility == "all_users")).all()
    member = session.exec(
        select(ProjectMember.project_id)
        .join(Project, Project.id == ProjectMember.project_id)
        .where(ProjectMember.user_id == user.id)
        .where(Project.workspace_id == user.workspace_id)
    ).all()
    return sorted({*public, *member})


def project_workspace(session: Session, project: Project):
    from app.models.workspace import Workspace
    return session.get(Workspace, project.workspace_id) if project.workspace_id else None


def require_not_demo_project(session: Session, project: Project, action: str) -> None:
    from app.auth.deps import require_not_demo
    workspace = project_workspace(session, project)
    if workspace is not None:
        require_not_demo(workspace, action)

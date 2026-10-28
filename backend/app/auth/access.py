"""
Project-level authorization.

Admins (managers) can do everything. Other users can read a project when it is
visible to all users or when they are a member; writing (uploading, deleting,
re-chunking) requires admin or an "editor" membership.
"""

from typing import List, Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.rag import Project, ProjectMember
from app.models.user import User, UserRole


def _membership(session: Session, project_id: int, user_id: int) -> Optional[ProjectMember]:
    return session.exec(
        select(ProjectMember)
        .where(ProjectMember.project_id == project_id)
        .where(ProjectMember.user_id == user_id)
    ).first()


def can_read_project(session: Session, user: User, project: Project) -> bool:
    if user.role == UserRole.ADMIN:
        return True
    if (project.visibility or "all_users") == "all_users":
        return True
    return _membership(session, project.id, user.id) is not None


def can_write_project(session: Session, user: User, project: Project) -> bool:
    if user.role == UserRole.ADMIN:
        return True
    member = _membership(session, project.id, user.id)
    return member is not None and member.role == "editor"


def require_project(session: Session, user: User, project_id: Optional[int], write: bool = False) -> Project:
    """Load a project the user may access, or raise 404/403.
    Unknown and forbidden projects both return 404 for readers so project IDs
    cannot be enumerated."""
    if project_id is None:
        raise HTTPException(status_code=400, detail="project_id is required")
    project = session.get(Project, project_id)
    if not project or not can_read_project(session, user, project):
        raise HTTPException(status_code=404, detail="Project not found")
    if write and not can_write_project(session, user, project):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this project")
    return project


def readable_project_ids(session: Session, user: User) -> Optional[List[int]]:
    """None means 'all projects' (admin)."""
    if user.role == UserRole.ADMIN:
        return None
    public = session.exec(select(Project.id).where(Project.visibility == "all_users")).all()
    member = session.exec(select(ProjectMember.project_id).where(ProjectMember.user_id == user.id)).all()
    return sorted({*public, *member})

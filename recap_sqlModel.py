from sqlmodel import SQLModel, Field
from sqlmodel import create_engine
from sqlalchemy.engine import Engine
from sqlmodel import Session
from typing import Generator
from contextlib import contextmanager
from sqlalchemy import UniqueConstraint


class Hero(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("name", "secret_name", "age"),
    )  # se applico nuovi constraints, gli item preesistenti non lo rispettano!
    id: int | None = Field(primary_key=True)
    name: str
    secret_name: str
    age: int | None = None


def get_engine(url: str) -> Engine:
    return create_engine(url=url)


@contextmanager
def get_session(engine: Engine):
    with Session(engine) as session:
        yield session


def create_db_and_tables(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def add_hero(session: Session, hero_to_add: Hero) -> Hero:
    try:
        session.add(hero_to_add)
        session.commit()
        session.refresh(hero_to_add)

        print(f"Hero with id {hero_to_add.id} added!")
    except ValueError:
        print("Hero exists already")


if __name__ == "__main__":
    hero = Hero(name="Bruce", secret_name="Batman", age=35)

    engine = get_engine("sqlite:///database.db")
    create_db_and_tables(engine)

    with get_session(engine) as session:
        add_hero(session, hero)

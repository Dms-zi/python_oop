from typing import Any, Optional


class Database:
    def __init__(self, name:str, connection: Optional[str] = None) -> None:
        self.name = name
        self.connection = connection


    def fetch(self, key: str) -> dict[str, Any]:
        # TODO: 리턴되는 값 코딩
        return {"key": key}
    

db: Optional[Database] = None


def initialize_database(connection: Optional[str] = None) -> None:
    global db
    db = Database(connection)


def get_database(connection : Optional[str] = None) ->  Database:
    global db
    if db is None:
        db = Database(connection)
    return db


class Query:
    def __init__(slef, database: Database, collection: str) -> None:
        pass

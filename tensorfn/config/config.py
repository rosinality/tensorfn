from pydantic import BaseModel


class Config(BaseModel):
    class Config:
        extra = "forbid"

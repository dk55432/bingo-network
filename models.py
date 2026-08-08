"""
Data models: a Player and the BingoCards that belong to them.

If you already have a players table/model elsewhere in your app, drop the
Player class here and just point BingoCard.player_id at your existing table
via the same foreign_key string (e.g. "yourplayertable.player_id").
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class Player(SQLModel, table=True):
    player_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    player_display_name: str

    cards: list["BingoCard"] = Relationship(back_populates="player")


class BingoCard(SQLModel, table=True):
    card_id: Optional[int] = Field(default=None, primary_key=True)

    player_id: str = Field(foreign_key="player.player_id", index=True)
    player: Optional[Player] = Relationship(back_populates="cards")

    # The confirmed 5x5 grid, e.g. [[3, 22, 31, 58, 61], ...].
    # The center cell is stored as null for the free space.
    grid: list[list[Optional[int]]] = Field(sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

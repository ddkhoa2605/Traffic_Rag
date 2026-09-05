from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractionConfig(BaseModel):
    parser_route: Literal["auto", "native", "vision"] = "auto"


class IssuingAuthority(BaseModel):
    authority_id: str = Field(pattern=r"^AUTH__[A-Z][A-Z0-9_]+$")
    title: str = Field(min_length=1)


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    document_number: str = Field(pattern=r"^\d+/\d{4}/QH\d+$")
    document_type: Literal["law"]
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    issued_date: date
    effective_from: date
    effective_to: date | None = None
    role: Literal["base"]
    parser_profile: Literal["law", "consolidated_law", "decree", "amendment", "circular", "technical_regulation"] = "law"
    reference_aliases: list[str] = Field(default_factory=list)
    issuing_authority: IssuingAuthority | None = None

    @field_validator("effective_to")
    @classmethod
    def effective_range(cls, value: date | None, info):
        start = info.data.get("effective_from")
        if value is not None and start is not None and value < start:
            raise ValueError("effective_to must not precede effective_from")
        return value


class SourceFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    document_id: str
    original_filename: str
    local_path: Path
    mime_type: Literal["application/pdf"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

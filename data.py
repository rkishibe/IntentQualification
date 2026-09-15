# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Company:
    operational_name: str = ""
    website: str = ""
    year_founded: Optional[float] = None
    address: Optional[dict] = None
    employee_count: Optional[float] = None
    revenue: Optional[float] = None
    primary_naics: Optional[dict] = None
    secondary_naics: Optional[dict] = None
    description: str = ""
    business_model: list = field(default_factory=list)
    core_offerings: list = field(default_factory=list)
    target_markets: list = field(default_factory=list)
    is_public: Optional[bool] = None

    def composite_text(self) -> str:
        """Rich text used for embedding. Robust to missing fields."""
        parts = [
            self.operational_name,
            self.description,
            " ".join(self.core_offerings or []),
            " ".join(self.target_markets or []),
            " ".join(self.business_model or []),
            (self.primary_naics or {}).get("label", "") if self.primary_naics else "",
            (self.secondary_naics or {}).get("label", "") if self.secondary_naics else "",
        ]
        return " . ".join(p for p in parts if p)

@dataclass
class Verdict:
    company: Company
    score: float
    matched: bool
    reason: str
    stage_reached: str
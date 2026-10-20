"""This week's collection days, from a schedule the user sets themselves.

PoiMae ships **no** collection calendar.  Collection days differ street by
street, and a fabricated calendar would send people to the curb on the wrong
morning - the same failure mode as a fabricated bin.  So the app asks the user
to enter the weekdays their own ward uses (with a link to the official lookup),
keeps that in the browser, and only does the arithmetic: what is due, on which
day, and how long until the next one.

The schedule never reaches a database.  The endpoint that calls this module is
stateless - the same JSON goes in and comes back out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

WEEKDAY_JA = ("月", "火", "水", "木", "金", "土", "日")


class ScheduleError(ValueError):
    """The user's schedule did not make sense."""


@dataclass(frozen=True)
class Due:
    day: date
    weekday: int            # ISO 1..7
    in_days: int
    categories: list[str]

    def as_dict(self) -> dict:
        return {
            "date": self.day.isoformat(),
            "weekday": WEEKDAY_JA[self.weekday - 1],
            "in_days": self.in_days,
            "label": "今日" if self.in_days == 0 else ("明日" if self.in_days == 1 else f"{self.in_days}日後"),
            "categories": self.categories,
        }


def validate_schedule(schedule: dict, allowed_categories: list[str]) -> dict[str, list[int]]:
    """Reject anything the app cannot honour instead of silently ignoring it."""
    if not isinstance(schedule, dict):
        raise ScheduleError("schedule must be an object of category -> [weekday]")
    cleaned: dict[str, list[int]] = {}
    for category, weekdays in schedule.items():
        if category not in allowed_categories:
            raise ScheduleError(
                f"unknown category {category!r}; "
                f"this municipality has {allowed_categories}"
            )
        if not isinstance(weekdays, (list, tuple)):
            raise ScheduleError(f"{category}: weekdays must be a list")
        seen: list[int] = []
        for weekday in weekdays:
            if not isinstance(weekday, int) or not 1 <= weekday <= 7:
                raise ScheduleError(
                    f"{category}: weekday must be 1..7 (Mon..Sun), got {weekday!r}"
                )
            if weekday not in seen:
                seen.append(weekday)
        if seen:
            cleaned[category] = sorted(seen)
    return cleaned


def week_plan(
    schedule: dict[str, list[int]],
    today: date,
    *,
    days: int = 7,
    allowed_categories: list[str] | None = None,
) -> dict:
    """The next `days` days of the user's schedule, plus the next day per bin."""
    if days < 1 or days > 31:
        raise ScheduleError("days must be 1..31")
    cleaned = (
        validate_schedule(schedule, allowed_categories)
        if allowed_categories is not None
        else validate_schedule(schedule, list(schedule))
    )

    due: list[Due] = []
    for offset in range(days):
        day = today + timedelta(days=offset)
        weekday = day.isoweekday()
        categories = sorted(
            category for category, dayset in cleaned.items() if weekday in dayset
        )
        if categories:
            due.append(Due(day=day, weekday=weekday, in_days=offset, categories=categories))

    next_by_category: dict[str, dict | None] = {}
    for category, dayset in cleaned.items():
        best: Due | None = None
        for offset in range(0, days):
            day = today + timedelta(days=offset)
            if day.isoweekday() in dayset:
                best = Due(day=day, weekday=day.isoweekday(), in_days=offset, categories=[category])
                break
        next_by_category[category] = best.as_dict() if best else None

    soonest = due[0] if due else None
    return {
        "today": today.isoformat(),
        "days": days,
        "schedule": cleaned,
        "due": [d.as_dict() for d in due],
        "next_by_category": next_by_category,
        "headline": (
            f"{soonest.as_dict()['label']}は"
            + "・".join(soonest.categories)
            if soonest
            else "この先の予定は未設定です"
        ),
        "unset": [c for c, w in cleaned.items() if not w] if cleaned else [],
    }

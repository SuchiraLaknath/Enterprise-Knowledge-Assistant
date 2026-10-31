"""Dummy enterprise data. All names and records here are invented for the assessment."""

EMPLOYEES = [
    {"name": "Asha Fernando", "department": "payments", "title": "Payments Platform Lead"},
    {"name": "Kavin Mendis", "department": "payments", "title": "Site Reliability Engineer"},
    {"name": "Tharushi Silva", "department": "security", "title": "Security Operations Analyst"},
    {"name": "Ruwan Jayasinghe", "department": "platform", "title": "Database Administrator"},
    {"name": "Dilini Wickrama", "department": "product", "title": "Mobile Product Owner"},
    {"name": "Nimal Gunawardena", "department": "hr-policy", "title": "HR Policy Partner"},
]

SERVICES = [
    {"name": "Card Switch", "owner": "Asha Fernando", "tier": "1", "on_call": "payments-primary"},
    {
        "name": "Payment Gateway API",
        "owner": "Kavin Mendis",
        "tier": "1",
        "on_call": "payments-primary",
    },
    {
        "name": "Real-Time Transfers",
        "owner": "Asha Fernando",
        "tier": "1",
        "on_call": "payments-primary",
    },
    {
        "name": "Merchant Settlement",
        "owner": "Kavin Mendis",
        "tier": "2",
        "on_call": "payments-secondary",
    },
    {
        "name": "Mobile Banking",
        "owner": "Dilini Wickrama",
        "tier": "1",
        "on_call": "digital-primary",
    },
    {
        "name": "ATM Network",
        "owner": "Ruwan Jayasinghe",
        "tier": "2",
        "on_call": "platform-primary",
    },
]

# `access_level` uses the same levels as documents, and the server filters on it by role.
INCIDENTS = [
    {
        "id": "LIVE-1041",
        "service": "Card Switch",
        "status": "open",
        "severity": "major",
        "summary": "Elevated declines on the switch-to-acquirer link.",
        "access_level": "internal",
    },
    {
        "id": "LIVE-1038",
        "service": "Merchant Settlement",
        "status": "resolved",
        "severity": "minor",
        "summary": "Settlement file delayed by forty minutes.",
        "access_level": "internal",
    },
    {
        "id": "LIVE-1036",
        "service": "Payment Gateway API",
        "status": "open",
        "severity": "major",
        "summary": "Suspected credential misuse under investigation.",
        "access_level": "confidential",
    },
    {
        "id": "LIVE-1029",
        "service": "Real-Time Transfers",
        "status": "resolved",
        "severity": "critical",
        "summary": "Regulator-notified outage with restricted findings.",
        "access_level": "restricted",
    },
]

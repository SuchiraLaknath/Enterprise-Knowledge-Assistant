# 0010. Demo users and one-click sign-in

Date: 2026-10-31
Status: Accepted

## Context

The brief allows hardcoded users. A demo must switch roles quickly, because role differences are a large part of what it shows.

## Decision

Three users are hardcoded in `api/auth.py`, and the UI signs in with a dropdown, without typing a password. The API still checks the password and issues a real JWT, and everything after sign-in uses that token, so the rest of the system behaves as it would with a real identity provider.

## Consequences

- Anyone who can open the page can be an administrator. So all ports bind to `127.0.0.1`, and the app must not be exposed to a network as it stands.
- Rate limits, audit entries and memory all apply per user, as they would in production.
- For production: replace `api/auth.py` with an identity provider such as Keycloak, which the brief also allows. Nothing downstream needs to change, because it reads only the verified `UserContext`.

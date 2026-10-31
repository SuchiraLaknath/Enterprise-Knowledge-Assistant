"""The request header that carries the verified role to the MCP server.

It lives here because both the client and the server need it, and the server runs apart from the
API, with no access to the rest of the application (rule 2).
"""

USER_ROLE_HEADER = "x-user-role"

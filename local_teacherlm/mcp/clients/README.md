# MCP Clients

External generators can be registered here during development. A presentation
generator should declare a manifest, permissions, supported artifact types, and
the transport TeacherLM should use to call it.

Connected tools must use TeacherLM-provided context or MCP tools. They must not
read the local SQLite database, uploaded files, secrets, or artifact folders
directly.


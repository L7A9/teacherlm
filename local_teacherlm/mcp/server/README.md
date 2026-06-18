# TeacherLM MCP Server

This folder holds the MCP-facing course-memory server for the desktop app.

The first implementation exposes the tool contract and a small Python module
that can be used by a real MCP transport later. External generators must not
read SQLite or the filesystem directly; they ask TeacherLM for scoped context
and write artifacts through TeacherLM APIs.


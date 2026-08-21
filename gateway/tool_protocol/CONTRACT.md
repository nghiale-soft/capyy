# Tool protocol contract

This boundary owns conversion between an upstream text protocol and a client’s
native tools. A call is valid only when its name was declared by the client and
its required arguments satisfy the declared JSON schema. Client tools are never
executed by the gateway. The compiler/repair path is private, bounded, and
never written to chat history.

Changes require regression coverage for: declared-tool validation, aliases,
empty required arguments, XML/DSML/JSON parsing, and final-answer detection.

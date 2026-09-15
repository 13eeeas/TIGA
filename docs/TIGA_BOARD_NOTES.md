# TIGA Board follow-up notes

Date: 2026-09-15 (Asia/Singapore)

## Kit template requirement from Shawn

Kit must follow HICA's folder structure, which Shawn identifies as the ISO-approved structure. Preserve the approved numbering, names, and hierarchy when deriving a reusable project template. The generic demo template has not been validated against HICA and does not establish acceptance.

The exact controlled reference is still to be confirmed: the live `F:\Shared\Projects\283 HICA` tree or a separate approved template. ISO approval is user-provided context, not independently verified in this session. No template implementation has been made yet.

## Hunt verification clarification

The update attempt failed because Git could not write `.git/FETCH_HEAD` (`Permission denied`). The session restricts writes to `.git`; this supports an execution-permission blocker, not a demonstrated Hunt application defect. An elevated retry was not attempted.

No listening service was observed on ports 7860/7862. A Hunt restart was not actually attempted, so startup failure and its cause were not established. The Projects browser flow was not exercised.

The CLI query returned no visible output within the observed tool window. No terminal result or underlying query error was captured; this does not prove Search is broken.

Indexing was incomplete in the captured snapshot. A single 0/h index throughput observation does not establish a persistent stall or explain the absent server. Full corpus completion is not required to test an already indexed project.

Next verification steps: retry the update with appropriate permissions while preserving local changes; start Hunt and inspect startup output; then run Search and the NUS BIZ3 Projects acceptance flow.

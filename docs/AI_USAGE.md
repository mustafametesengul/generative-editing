# AI Usage Disclosure

This submission was created with AI assistance.

## Division of work

The human author chose the problem framing, weather-editing domain, goals, constraints, and final design direction. The author reviewed the results, challenged unclear or unrealistic proposals, and made the final decisions about what to include.

GitHub Copilot was used to:

- expand and compare technical approaches;
- draft and simplify parts of the documentation;
- help write and revise some prototype and test code;
- review the repository for gaps, inconsistencies, and unsupported claims;
- check model-card facts and clarify prototype versus production scope;
- run formatting, linting, tests, and documentation checks.

In short, the core ideas and decisions were human-led. AI helped turn them into detailed documents and working code, and also acted as a reviewer.

## How suggestions were checked

AI output was not accepted automatically. Claims were compared with model documentation and the repository implementation. Code changes were checked with Ruff and the automated test suite. The human author remains responsible for the final submission and its technical claims.

## AI models used by the prototype

This disclosure is separate from the models evaluated by the solution itself. The demo uses open-weight perception and image-editing models, including SegFormer, Depth Anything V2, and FLUX.2 Klein. Their identifiers, roles, and limitations are documented in the task documents and source code; no model weights are included in the repository.

## Activity summary

| Stage | Human contribution | AI assistance |
| --- | --- | --- |
| Problem definition | Chose the domain, edit contract, priorities, and constraints | Explored alternatives and trade-offs |
| System design | Selected the direction and challenged weak assumptions | Drafted details, failure modes, metrics, and deployment options |
| Prototype | Reviewed behavior and accepted final changes | Helped implement components, tests, and fixes |
| Documentation | Set the message and approved the final wording | Drafted, shortened, and checked consistency |
| Verification | Judged whether the result met the assignment | Ran linting, tests, link checks, and repository audits |

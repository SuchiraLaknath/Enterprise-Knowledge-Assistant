"""Turn the findings of many sub-agents into one report.

The LLM decides which findings belong together, and code does everything else: it counts the
documents, finds the dates and checks the finding ids. An answer can then state "8 incidents"
because 8 documents were counted, not because a model said so. If no model can group the
findings, their own labels group them instead.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.evidence_formatting import format_document_tag, make_prompt_safe
from agent.llm_gateway import AllModelsFailedError
from agent.prompts.research import AGGREGATOR_REQUEST_TEMPLATE, AGGREGATOR_SYSTEM_PROMPT
from agent.schemas import Finding, ResearchTheme, ThemeDraft, ThemeGrouping

# Grouping decides what the final answer says, so it shares the strong route.
AGGREGATOR_ROUTE = "response"

DEGRADED_MODEL_FALLBACK = "research_aggregator_used_fallback_model"
DEGRADED_GROUPED_BY_LABEL = "research_grouped_by_label"

UNGROUPED_THEME_NAME = "Other findings"
UNGROUPED_THEME_SUMMARY = "Findings that did not fit any of the themes above."
LABEL_THEME_SUMMARY = "Findings that sub-agents labelled in the same way."


def group_by_label(findings: list[Finding]) -> list[ThemeDraft]:
    """Group findings by their own labels, for when no model can group them.

    Different wordings of one cause stay apart, which is why this is only the fallback.
    """
    drafts_by_label: dict[str, ThemeDraft] = {}
    for finding in findings:
        normalized_label = finding.theme_label.strip().lower()
        if normalized_label not in drafts_by_label:
            drafts_by_label[normalized_label] = ThemeDraft(
                name=finding.theme_label.strip(), summary=LABEL_THEME_SUMMARY, finding_ids=[]
            )
        drafts_by_label[normalized_label].finding_ids.append(finding.finding_id)
    return list(drafts_by_label.values())


async def propose_themes(
    question: str, findings: list[Finding], dependencies: AgentDependencies
) -> tuple[list[ThemeDraft], list[str]]:
    """Ask the LLM to group the findings, or group them by label if no model can."""
    # Rule 5: findings come from documents, so they stay inside document tags here too.
    finding_tags = [
        format_document_tag(finding.finding_id, finding.title, finding.theme_label, finding.claim)
        for finding in findings
    ]
    request_text = AGGREGATOR_REQUEST_TEMPLATE.format(
        question=make_prompt_safe(question), findings="\n\n".join(finding_tags)
    )
    messages: list[BaseMessage] = [
        SystemMessage(AGGREGATOR_SYSTEM_PROMPT),
        HumanMessage(request_text),
    ]
    try:
        grouping_result = await dependencies.llm.generate_structured(
            AGGREGATOR_ROUTE, messages, ThemeGrouping
        )
    except AllModelsFailedError:
        return group_by_label(findings), [DEGRADED_GROUPED_BY_LABEL]

    degraded = grouping_result.describe_fallbacks(DEGRADED_MODEL_FALLBACK)
    return grouping_result.value.themes, degraded


def _build_theme(name: str, summary: str, findings: list[Finding]) -> ResearchTheme:
    """Calculate a theme's counts and dates from its findings, of which there is at least one."""
    created_dates = sorted(finding.created_date for finding in findings)
    return ResearchTheme(
        name=name,
        summary=summary,
        document_count=len({finding.doc_id for finding in findings}),
        first_date=created_dates[0],
        last_date=created_dates[-1],
        findings=findings,
    )


def build_themes(theme_drafts: list[ThemeDraft], findings: list[Finding]) -> list[ResearchTheme]:
    """Turn proposed groups into themes, largest first.

    Ids a model made up are ignored, and a finding it placed twice counts once. Any finding it
    left out goes in a final theme, so that evidence is never lost in silence.
    """
    findings_by_id = {finding.finding_id: finding for finding in findings}
    placed_finding_ids: set[str] = set()
    themes = []
    for theme_draft in theme_drafts:
        theme_findings = []
        for finding_id in theme_draft.finding_ids:
            if finding_id in findings_by_id and finding_id not in placed_finding_ids:
                theme_findings.append(findings_by_id[finding_id])
                placed_finding_ids.add(finding_id)
        if theme_findings:
            themes.append(_build_theme(theme_draft.name, theme_draft.summary, theme_findings))

    themes.sort(key=lambda theme: theme.document_count, reverse=True)

    # Added after sorting, so that leftovers stay last however many there are.
    ungrouped_findings = [
        finding for finding in findings if finding.finding_id not in placed_finding_ids
    ]
    if ungrouped_findings:
        themes.append(
            _build_theme(UNGROUPED_THEME_NAME, UNGROUPED_THEME_SUMMARY, ungrouped_findings)
        )
    return themes

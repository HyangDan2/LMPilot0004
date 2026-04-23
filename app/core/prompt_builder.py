from __future__ import annotations


def build_user_prompt(import_context: str, current_code: str, bug_report: str) -> str:
    return (
        '[IMPORT_LAYER]\n'
        f'{import_context.strip()}\n\n'
        '[CURRENT_CODE]\n'
        f'{current_code.strip()}\n\n'
        '[ISSUE]\n'
        f'{bug_report.strip()}\n\n'
        '[TASK]\n'
        'Fix the issue and return the corrected code. '\
        'Preserve intended features when possible. '\
        'When the issue description is incomplete, infer the most likely safe fix.\n\n'
        '[OUTPUT_FORMAT]\n'
        '1) Brief diagnosis\n'
        '2) Corrected code\n'
        '3) Key changes summary\n'
    )

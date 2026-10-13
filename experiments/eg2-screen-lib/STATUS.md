# eg2-screen-lib status (2026-10-13, pm)

- survey.py: 60 captures bot17, all start_norah (stuck bot).
- Dismiss x2 (Norah popup -> afk_popup, replicated). Proposal:
  start_norah --click(1050,100)--> afk_popup (no out-edges in DB today).
- Confirm (741,676) -> shelter_view. DB claims afk_popup -> world_map;
  OBSERVED shelter_view. Edge stale or transition skipped; flagged.
- walker.py: afk_popup ESC-resistant (stop, correct). From shelter_view the
  walker 2-cycled shelter<->exit_dialog 12x (no loop detection — fixed:
  stop on repeat edge). 2 NEW edges observed (neither in DB):
  shelter_view --esc--> exit_dialog ; exit_dialog --esc--> shelter_view
  (DB knew only click-exit_cancel out of exit_dialog).
- Audit: 5 unmatchable screens (bag/mission/messages/hero/region_overview),
  9 no-out-edge screens, descriptions complete.
- Bot17 parked on shelter_view (post exit_dialog cancel).
- EG2 :8091 + mmproj up. Qwen3.8-27B :8080 now completion+multimodal (6.6s/describe).

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

## Tutorial attempt 2026-10-13 (hero_details, "Upgrade Heroes" banner)
- up-arrow (1085,315): no change. Skill button (1020,730): no change.
  fist icon (1090,625): no change. Hero Talent (855,730): no change.
- "+" tap hit SKIP's hitbox -> "Skip Tutorial? Cancel/Skip" dialog
  (identify=None: genuine unknown-screen candidate, saved tutorial_skip_dialog.png).
  Cancelled back to hero_details, tutorial intact.
- Qwen suggested SKIP or upgrade target (reasoning cut by token budget).
- Stopped blind-tapping after 4 misses. Next candidates: grenade icon,
  precise EXP "+" coords, or Qwen with bigger budget.

## Fixed-banner template check 2026-10-13
- 10% patch of hero-upgrade banner (t00 rect 677,349,475,135 -> patch 840,395,150,45):
  banner frames 1.000 / all others <=0.492 (skip dialogs 0.49, sky 0.21).
  Threshold 0.75, huge margin. ms on CPU, no GPU.
- Doctrine: fixed-location banners -> template patch (this); unknown overlays ->
  contour localize + EG2 crop-name; genuinely new screens -> Qwen-name + bank.
  Each layer handles what the layer below cannot see.

## Systemd cutover 2026-10-13
- Models moved to /usr/local/devel/models/embedding/ (text 310MB + mmproj 555MB).
- /etc/llama-eg2.config + llama-eg2.service created, enabled, active.
- Gotcha: $VAR inside EXTRA_OPTIONS does NOT expand from EnvironmentFile
  (mmproj failed to load as literal '$MMPROJ_FILE'); path inlined.
  NOTE: /etc/llama-vision.config has the same latent bug.
- Verified: red-query vs red-image cos 0.780 under unit (was 0.749-0.78 manual).
  Identical vectors. Manual server retired.

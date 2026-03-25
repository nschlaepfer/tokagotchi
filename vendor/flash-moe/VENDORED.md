# Vendored `flash-moe`

- Upstream: `https://github.com/Anemll/flash-moe`
- Branch: `iOS-App`
- Commit: `9d1d602bceca11b2e1b9fd6cf89057858cfbe6e4`
- Vendored on: `2026-03-24`

## Scope

This snapshot is vendored so we can build a local macOS app wrapper around the upstream Flash-MoE C/Metal engine without taking a live Git dependency.

## Local adjustments

- Kept the upstream source tree under `vendor/flash-moe/`.
- Patched the Apple project to target native macOS builds instead of the original iOS-first signing setup.
- Switched model storage to `Application Support/FlashMoE/Models` on macOS instead of the iOS Documents directory.
- Added `scripts/vendor_flash_moe.sh` to refresh the vendored snapshot.

## Refresh

Run:

```bash
bash scripts/vendor_flash_moe.sh
```

Then re-apply any local project patches that should stay on top of upstream changes.

## Notes

- I did not find a `LICENSE` file in the vendored upstream snapshot. Verify redistribution terms before shipping this outside internal use.
- Local verification in this workspace reached Xcode's Metal compilation step, but full builds are currently blocked here by a missing local Metal toolchain component.

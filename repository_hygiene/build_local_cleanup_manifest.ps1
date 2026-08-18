$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $workspace
$entries = @()

function Add-CleanupEntry {
    param(
        [string]$Path,
        [string]$Classification,
        [string]$Action,
        [string]$Reason,
        [string]$Recoverability
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) {
        $bytes = (Get-ChildItem -LiteralPath $item.FullName -Recurse -File -Force `
            -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    } else {
        $bytes = $item.Length
    }
    $script:entries += [pscustomobject]@{
        path = $Path
        absolute_path = $item.FullName
        bytes = [int64]$bytes
        classification = $Classification
        action = $Action
        reason = $Reason
        recoverability = $Recoverability
    }
}

$directoryTargets = @(
    ".pytest_cache", ".codex_tmp", ".codex_no_holdout_20260741",
    ".codex_training_diagnostics_20260741", "tmp", "diagnostic_analysis",
    "data/_codex_generator_smoke", "data/_codex_generator_smoke2",
    "data/_codex_generator_smoke3", "data/rl_runs",
    "data/synthetic_diagnostics", "logs", "figures", "paper_revision",
    "CUHK_SZ_Vine_RL_Paper_revised"
)
foreach ($path in $directoryTargets) {
    Add-CleanupEntry $path "generated_or_obsolete" "delete_local" `
        "Cache, smoke output, superseded diagnostic/run, or superseded manuscript tree." `
        "Regenerable or superseded by frozen evidence/latest manuscript."
}

$fileTargets = @(
    "data/pretrain_returns.qs", "data/benchmark_results.RData",
    "data/eu_backtest_result.RData", "data/evaluation_comparison_default.RData",
    "data/evaluation_logs_default.csv", "data/evaluation_results.RData",
    "data/evaluation_summary_default.csv", "data/marginal_results.RData",
    "data/nn_backtest_result.RData", "data/vine_fit.RData",
    "CUHK_SZ_Vine_RL_Paper_revised.zip",
    "CUHK_SZ_Vine_RL_Paper_revised_causal_v1.zip", "paper_revision.zip"
)
foreach ($path in $fileTargets) {
    Add-CleanupEntry $path "generated_or_duplicate" "delete_local" `
        "Obsolete pre-schema output or duplicate archive." `
        "Regenerable; final evidence/manuscript retained."
}

$manuscriptTargets = @(
    "manuscript_revision_causal_v1/main.aux",
    "manuscript_revision_causal_v1/main.blg",
    "manuscript_revision_causal_v1/main.log",
    "manuscript_revision_causal_v1/main.out",
    "manuscript_revision_causal_v1/figures/lstm_ddpg_architecture.pdf",
    "manuscript_revision_causal_v1/figures/training_strategy_old.pdf"
)
foreach ($path in $manuscriptTargets) {
    Add-CleanupEntry $path "build_or_superseded_figure" "delete_local" `
        "LaTeX intermediate or unreferenced obsolete figure." `
        "Regenerable from source."
}

$canonical = @(
    "analysis_outputs/oos_v4_verified_770d2944",
    "analysis_outputs/post_hoc_compressed_vine_benchmark_reconciliation_v1",
    "frozen_releases/final_evidence/causal_results_v2_v3_v4_plot_runtime_v1.tar.gz",
    "manuscript_revision_causal_v1", "data/portfolio_A_5assets_2007.csv",
    "data/portfolio_B_7assets_2013.csv", "data/portfolio_C_8assets_2015.csv"
)
foreach ($path in $canonical) {
    Add-CleanupEntry $path "canonical" "retain" `
        "Current evidence, source manuscript, or raw input." "Authoritative local copy."
}

$provenance = @(
    "frozen_releases/post_holdout_explanatory_ablation_v2",
    "frozen_releases/post_holdout_secondary_plan_v1"
)
foreach ($path in $provenance) {
    Add-CleanupEntry $path "immutable_provenance" "retain" `
        "Frozen scientific registration/release." "Do not edit."
}

$output = Join-Path $PSScriptRoot "local_cleanup_manifest.csv"
$entries | Export-Csv -NoTypeInformation -Encoding UTF8 $output
$entries | Group-Object action | ForEach-Object {
    [pscustomobject]@{
        action = $_.Name
        items = $_.Count
        bytes = ($_.Group | Measure-Object bytes -Sum).Sum
    }
}

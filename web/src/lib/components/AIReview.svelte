<script lang="ts">
	let { review }: { review: string } = $props();
</script>

<div class="ai-review">
	<div class="review-header">
		<span class="review-icon">🤖</span>
		<span class="review-title">AI Review</span>
	</div>
	<div class="review-body">{review}</div>
</div>

<style>
	.ai-review {
		margin-top: 10px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-left: 3px solid var(--purple);
		border-radius: 0 6px 6px 0;
		overflow: hidden;
	}

	.review-header {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 8px 12px;
		background: rgba(163, 113, 247, 0.05);
		border-bottom: 1px solid var(--border);
	}

	.review-icon {
		font-size: 14px;
	}

	.review-title {
		font-size: 12px;
		font-weight: 600;
		color: var(--purple);
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.review-body {
		padding: 12px;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>

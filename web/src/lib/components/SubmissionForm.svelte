<script lang="ts">
	import { enhance } from '$app/forms';
	import AIReview from './AIReview.svelte';

	let {
		milestoneId,
		languages,
		submissions,
		hasApiKey
	}: {
		milestoneId: number;
		languages: Array<{ language: string; recommended: number }>;
		submissions: Array<{
			id: number;
			content: string;
			language: string | null;
			status: string;
			createdAt: string;
			review: string | null;
		}>;
		hasApiKey: boolean;
	} = $props();

	let submitting = $state(false);
	let reviewingId = $state<number | null>(null);
	let reviewResult = $state<string | null>(null);
	let reviewError = $state<string | null>(null);
	let guideLoading = $state(false);
	let guideResult = $state<string | null>(null);
	let guideError = $state<string | null>(null);
	let guideQuestion = $state('');
	let showGuide = $state(false);

	async function requestReview(submissionId: number) {
		reviewingId = submissionId;
		reviewResult = null;
		reviewError = null;

		try {
			const res = await fetch('/api/ai/review', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ submissionId })
			});
			const data = await res.json();
			if (!res.ok) {
				reviewError = data.error || 'Review failed';
			} else {
				reviewResult = data.review;
			}
		} catch (err) {
			reviewError = 'Failed to connect to AI service';
		} finally {
			reviewingId = null;
		}
	}

	async function askGuide() {
		if (!guideQuestion.trim()) return;
		guideLoading = true;
		guideResult = null;
		guideError = null;

		try {
			const res = await fetch('/api/ai/guide', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ milestoneId, question: guideQuestion })
			});
			const data = await res.json();
			if (!res.ok) {
				guideError = data.error || 'Guidance failed';
			} else {
				guideResult = data.guidance;
			}
		} catch (err) {
			guideError = 'Failed to connect to AI service';
		} finally {
			guideLoading = false;
		}
	}

	function formatDate(iso: string) {
		return new Date(iso).toLocaleDateString('en-US', {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
</script>

<div class="submission-section">
	<!-- Previous Submissions -->
	{#if submissions.length > 0}
		<div class="previous-submissions">
			<div class="section-label">Previous Submissions</div>
			{#each submissions as sub}
				<div class="prev-submission">
					<div class="prev-header">
						<span class="prev-date">{formatDate(sub.createdAt)}</span>
						{#if sub.language}
							<span class="prev-lang">{sub.language}</span>
						{/if}
						<span class="prev-status" class:reviewed={sub.status === 'reviewed'}>
							{sub.status}
						</span>
					</div>
					<pre class="prev-code"><code>{sub.content.length > 300 ? sub.content.slice(0, 300) + '...' : sub.content}</code></pre>

					{#if sub.review}
						<AIReview review={sub.review} />
					{:else if hasApiKey}
						<button
							class="btn-review"
							onclick={() => requestReview(sub.id)}
							disabled={reviewingId === sub.id}
						>
							{reviewingId === sub.id ? 'Reviewing...' : 'Request AI Review'}
						</button>
					{/if}
				</div>
			{/each}
		</div>
	{/if}

	{#if reviewResult}
		<AIReview review={reviewResult} />
	{/if}
	{#if reviewError}
		<div class="error-msg">{reviewError}</div>
	{/if}

	<!-- Submit Work Form -->
	<div class="submit-form">
		<div class="section-label">Submit Work</div>
		<form
			method="POST"
			action="?/submitWork"
			use:enhance={() => {
				submitting = true;
				return async ({ update }) => {
					submitting = false;
					await update();
				};
			}}
		>
			<input type="hidden" name="milestoneId" value={milestoneId} />

			<div class="form-row">
				<label for="language-{milestoneId}">Language</label>
				<select id="language-{milestoneId}" name="language">
					<option value="">Select language</option>
					{#each languages as lang}
						<option value={lang.language}>
							{lang.language}{lang.recommended ? ' (recommended)' : ''}
						</option>
					{/each}
				</select>
			</div>

			<div class="form-row">
				<label for="content-{milestoneId}">Code / Text</label>
				<textarea
					id="content-{milestoneId}"
					name="content"
					placeholder="Paste your code or write your submission here..."
					rows="10"
					required
				></textarea>
			</div>

			<div class="form-actions">
				<button type="submit" class="btn-submit" disabled={submitting}>
					{submitting ? 'Submitting...' : 'Submit'}
				</button>

				{#if hasApiKey}
					<button type="button" class="btn-guide" onclick={() => (showGuide = !showGuide)}>
						Ask AI for Guidance
					</button>
				{:else}
					<span class="no-key-hint">
						<a href="/settings">Add API key</a> for AI features
					</span>
				{/if}
			</div>
		</form>
	</div>

	<!-- AI Guidance -->
	{#if showGuide}
		<div class="guide-section">
			<div class="section-label">Ask AI for Guidance</div>
			<textarea
				bind:value={guideQuestion}
				placeholder="What are you stuck on? Ask for hints without getting the answer..."
				rows="3"
			></textarea>
			<button class="btn-guide-send" onclick={askGuide} disabled={guideLoading || !guideQuestion.trim()}>
				{guideLoading ? 'Thinking...' : 'Ask'}
			</button>
			{#if guideResult}
				<div class="guide-result">{guideResult}</div>
			{/if}
			{#if guideError}
				<div class="error-msg">{guideError}</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.submission-section {
		margin-top: 16px;
		padding-top: 16px;
		border-top: 1px solid var(--border);
	}

	.section-label {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 10px;
	}

	.previous-submissions {
		margin-bottom: 20px;
	}

	.prev-submission {
		margin-bottom: 12px;
		padding: 10px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	.prev-header {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}

	.prev-date {
		font-size: 12px;
		color: var(--text-muted);
	}

	.prev-lang {
		padding: 1px 6px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 3px;
		font-size: 11px;
		color: var(--text-muted);
	}

	.prev-status {
		padding: 1px 6px;
		background: rgba(210, 153, 34, 0.15);
		color: var(--orange);
		border-radius: 3px;
		font-size: 11px;
		text-transform: capitalize;
	}

	.prev-status.reviewed {
		background: rgba(63, 185, 80, 0.15);
		color: var(--accent-bright);
	}

	.prev-code {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 8px;
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		line-height: 1.5;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.btn-review {
		margin-top: 8px;
		padding: 4px 10px;
		background: rgba(163, 113, 247, 0.1);
		border: 1px solid rgba(163, 113, 247, 0.3);
		border-radius: 4px;
		color: var(--purple);
		cursor: pointer;
		font-size: 12px;
		font-family: inherit;
	}

	.btn-review:hover:not(:disabled) {
		background: rgba(163, 113, 247, 0.2);
	}

	.btn-review:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.error-msg {
		padding: 8px 12px;
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.3);
		border-radius: 4px;
		color: var(--red);
		font-size: 13px;
		margin-top: 8px;
	}

	.submit-form {
		margin-top: 8px;
	}

	.form-row {
		margin-bottom: 12px;
	}

	.form-row label {
		display: block;
		font-size: 12px;
		color: var(--text-muted);
		margin-bottom: 4px;
	}

	.form-row select {
		padding: 6px 10px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text-primary);
		font-family: inherit;
		font-size: 13px;
	}

	.form-row textarea {
		width: 100%;
		padding: 10px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-primary);
		font-family: 'JetBrains Mono', monospace;
		font-size: 13px;
		line-height: 1.5;
		resize: vertical;
	}

	.form-row textarea:focus,
	.form-row select:focus {
		outline: none;
		border-color: var(--blue);
	}

	.form-actions {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.btn-submit {
		padding: 6px 14px;
		background: var(--accent);
		border: none;
		border-radius: 4px;
		color: white;
		cursor: pointer;
		font-size: 13px;
		font-weight: 500;
		font-family: inherit;
	}

	.btn-submit:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	.btn-submit:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-guide {
		padding: 6px 14px;
		background: none;
		border: 1px solid var(--purple);
		border-radius: 4px;
		color: var(--purple);
		cursor: pointer;
		font-size: 13px;
		font-family: inherit;
	}

	.btn-guide:hover {
		background: rgba(163, 113, 247, 0.1);
	}

	.no-key-hint {
		font-size: 12px;
		color: var(--text-muted);
	}

	.no-key-hint a {
		color: var(--blue);
	}

	.guide-section {
		margin-top: 16px;
		padding: 12px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
	}

	.guide-section textarea {
		width: 100%;
		padding: 8px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text-primary);
		font-family: inherit;
		font-size: 13px;
		resize: vertical;
		margin-bottom: 8px;
	}

	.guide-section textarea:focus {
		outline: none;
		border-color: var(--blue);
	}

	.btn-guide-send {
		padding: 6px 14px;
		background: var(--purple);
		border: none;
		border-radius: 4px;
		color: white;
		cursor: pointer;
		font-size: 13px;
		font-family: inherit;
	}

	.btn-guide-send:hover:not(:disabled) {
		opacity: 0.9;
	}

	.btn-guide-send:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.guide-result {
		margin-top: 12px;
		padding: 12px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-left: 3px solid var(--purple);
		border-radius: 0 6px 6px 0;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>

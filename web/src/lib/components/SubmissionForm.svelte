<script lang="ts">
	import { enhance } from '$app/forms';
	import AIReview from './AIReview.svelte';

	let {
		milestoneId,
		submissions,
		hasApiKey
	}: {
		milestoneId: number;
		submissions: Array<{
			id: number;
			fileName: string;
			fileSize: number;
			status: string;
			createdAt: string;
			review: string | null;
		}>;
		hasApiKey: boolean;
	} = $props();

	let submitting = $state(false);
	let reviewingId = $state<number | null>(null);
	let reviewResults = $state<Map<number, string>>(new Map());
	let reviewError = $state<string | null>(null);
	let selectedFile = $state<File | null>(null);
	let fileError = $state<string | null>(null);
	let guideLoading = $state(false);
	let guideResult = $state<string | null>(null);
	let guideError = $state<string | null>(null);
	let guideQuestion = $state('');
	let showGuide = $state(false);

	const MAX_SIZE = 5 * 1024 * 1024;

	function handleFileChange(e: Event) {
		const input = e.target as HTMLInputElement;
		const file = input.files?.[0] || null;
		fileError = null;

		if (file) {
			if (!file.name.endsWith('.zip')) {
				fileError = 'Only .zip files are allowed';
				selectedFile = null;
				input.value = '';
				return;
			}
			if (file.size > MAX_SIZE) {
				fileError = `File too large (${formatSize(file.size)}). Max 5MB.`;
				selectedFile = null;
				input.value = '';
				return;
			}
		}
		selectedFile = file;
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return bytes + ' B';
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
		return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	}

	async function requestReview(submissionId: number) {
		reviewingId = submissionId;
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
				reviewResults.set(submissionId, data.review);
				reviewResults = new Map(reviewResults);
			}
		} catch {
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
		} catch {
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
						<span class="file-info">📦 {sub.fileName} ({formatSize(sub.fileSize)})</span>
						<span class="prev-status" class:reviewed={sub.status === 'reviewed'}>
							{sub.status}
						</span>
					</div>

					{#if sub.review}
						<AIReview review={sub.review} />
					{:else if reviewResults.get(sub.id)}
						<AIReview review={reviewResults.get(sub.id) || ''} />
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
			{#if reviewError}
				<div class="error-msg">{reviewError}</div>
			{/if}
		</div>
	{/if}

	<!-- Submit Work Form -->
	<div class="submit-form">
		<div class="section-label">Submit Work</div>
		<form
			method="POST"
			action="?/submitWork"
			enctype="multipart/form-data"
			use:enhance={() => {
				submitting = true;
				return async ({ update }) => {
					submitting = false;
					selectedFile = null;
					await update();
				};
			}}
		>
			<input type="hidden" name="milestoneId" value={milestoneId} />

			<div class="file-upload">
				<label for="file-{milestoneId}" class="file-label">
					<span class="file-icon">📁</span>
					{#if selectedFile}
						<span class="file-name">{selectedFile.name}</span>
						<span class="file-size">{formatSize(selectedFile.size)}</span>
					{:else}
						<span class="file-placeholder">Choose a .zip file (max 5MB)</span>
					{/if}
				</label>
				<input
					type="file"
					id="file-{milestoneId}"
					name="file"
					accept=".zip"
					onchange={handleFileChange}
					class="file-input"
				/>
			</div>

			{#if fileError}
				<div class="error-msg">{fileError}</div>
			{/if}

			<div class="form-actions">
				<button type="submit" class="btn-submit" disabled={submitting || !selectedFile}>
					{submitting ? 'Uploading...' : 'Submit'}
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
		flex-wrap: wrap;
	}

	.prev-date {
		font-size: 12px;
		color: var(--text-muted);
	}

	.file-info {
		font-size: 12px;
		color: var(--text-secondary);
		font-family: 'JetBrains Mono', monospace;
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

	.file-upload {
		position: relative;
		margin-bottom: 12px;
	}

	.file-input {
		position: absolute;
		inset: 0;
		opacity: 0;
		cursor: pointer;
	}

	.file-label {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 12px 14px;
		background: var(--bg-dark);
		border: 2px dashed var(--border);
		border-radius: 6px;
		cursor: pointer;
		transition: border-color 0.15s;
	}

	.file-label:hover {
		border-color: var(--blue);
	}

	.file-icon {
		font-size: 18px;
	}

	.file-placeholder {
		font-size: 13px;
		color: var(--text-muted);
	}

	.file-name {
		font-size: 13px;
		color: var(--text-primary);
		font-weight: 500;
	}

	.file-size {
		font-size: 12px;
		color: var(--text-muted);
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

<script lang="ts">
	import { enhance } from '$app/forms';
	import AIReview from './AIReview.svelte';
	import FileTreeBrowser from './FileTreeBrowser.svelte';

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
	let dragOver = $state(false);

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

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const file = e.dataTransfer?.files?.[0];
		if (!file) return;
		fileError = null;

		if (!file.name.endsWith('.zip')) {
			fileError = 'Only .zip files are allowed';
			return;
		}
		if (file.size > MAX_SIZE) {
			fileError = `File too large (${formatSize(file.size)}). Max 5MB.`;
			return;
		}

		selectedFile = file;
		// Update the file input
		const input = document.getElementById(`file-${milestoneId}`) as HTMLInputElement;
		if (input) {
			const dt = new DataTransfer();
			dt.items.add(file);
			input.files = dt.files;
		}
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
	<!-- Upload Form -->
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

		<div
			class="upload-zone"
			class:drag-over={dragOver}
			class:has-file={selectedFile}
			role="button"
			tabindex="-1"
			ondragover={(e) => { e.preventDefault(); dragOver = true; }}
			ondragleave={() => { dragOver = false; }}
			ondrop={handleDrop}
		>
			<label for="file-{milestoneId}" class="upload-label">
				{#if selectedFile}
					<div class="upload-selected">
						<svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor" class="upload-icon-selected">
							<path d="M3.5 9.75A.75.75 0 0 1 4.25 9h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.5 9.75Zm0-3.5A.75.75 0 0 1 4.25 5.5h6.5a.75.75 0 0 1 0 1.5h-6.5a.75.75 0 0 1-.75-.75ZM3.5 3A.75.75 0 0 1 4.25 2.25h6.5a.75.75 0 0 1 0 1.5h-6.5A.75.75 0 0 1 3.5 3Z"/>
						</svg>
						<div class="selected-info">
							<span class="selected-name">{selectedFile.name}</span>
							<span class="selected-size">{formatSize(selectedFile.size)}</span>
						</div>
						<span class="selected-check">
							<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"/></svg>
						</span>
					</div>
				{:else}
					<div class="upload-empty">
						<svg width="24" height="24" viewBox="0 0 16 16" fill="currentColor" class="upload-icon">
							<path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14ZM11.78 4.72a.749.749 0 1 1-1.06 1.06L8.75 3.811V9.5a.75.75 0 0 1-1.5 0V3.811L5.28 5.78a.749.749 0 1 1-1.06-1.06l3.25-3.25a.749.749 0 0 1 1.06 0l3.25 3.25Z"/>
						</svg>
						<span class="upload-text">Drop your .zip file here or click to browse</span>
						<span class="upload-hint">Maximum file size: 5MB</span>
					</div>
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
				{#if submitting}
					<span class="spinner"></span>
					Uploading...
				{:else}
					<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14ZM11.78 4.72a.749.749 0 1 1-1.06 1.06L8.75 3.811V9.5a.75.75 0 0 1-1.5 0V3.811L5.28 5.78a.749.749 0 1 1-1.06-1.06l3.25-3.25a.749.749 0 0 1 1.06 0l3.25 3.25Z"/></svg>
					Submit
				{/if}
			</button>

			{#if hasApiKey}
				<button type="button" class="btn-guide" onclick={() => (showGuide = !showGuide)}>
					<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.92 6.085h.001a.749.749 0 1 1-1.342-.67c.169-.339.436-.701.849-.977C6.845 4.16 7.369 4 8 4a2.756 2.756 0 0 1 1.637.525c.503.377.863.965.863 1.725 0 .448-.115.83-.329 1.15-.205.307-.478.513-.708.662-.04.027-.08.052-.118.076-.428.27-.525.39-.525.61a.75.75 0 0 1-1.5 0c0-.76.478-1.204.964-1.51.162-.103.283-.184.378-.252.118-.084.182-.146.222-.207a.72.72 0 0 0 .116-.378c0-.274-.12-.452-.294-.582A1.264 1.264 0 0 0 8 5.5c-.39 0-.622.1-.784.206a1.174 1.174 0 0 0-.376.39l-.04.075Zm.94 5.165a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5Z"/></svg>
					Ask AI for Guidance
				</button>
			{:else}
				<span class="no-key-hint">
					<a href="/settings">Add API key</a> for AI features
				</span>
			{/if}
		</div>
	</form>

	<!-- AI Guidance -->
	{#if showGuide}
		<div class="guide-section">
			<textarea
				bind:value={guideQuestion}
				placeholder="What are you stuck on? Ask for hints without getting the answer..."
				rows="3"
			></textarea>
			<button class="btn-guide-send" onclick={askGuide} disabled={guideLoading || !guideQuestion.trim()}>
				{#if guideLoading}
					<span class="spinner"></span>
					Thinking...
				{:else}
					Ask
				{/if}
			</button>
			{#if guideResult}
				<div class="guide-result">{guideResult}</div>
			{/if}
			{#if guideError}
				<div class="error-msg">{guideError}</div>
			{/if}
		</div>
	{/if}

	<!-- Previous Submissions -->
	{#if submissions.length > 0}
		<div class="previous-submissions">
			<div class="submissions-label">Previous Submissions</div>
			{#each submissions as sub}
				<div class="prev-submission">
					<div class="prev-header">
						<div class="prev-info">
							<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" class="prev-file-icon">
								<path d="M3.5 1.75v11.5c0 .09.048.173.126.217a.75.75 0 0 1-.752 1.298A1.748 1.748 0 0 1 2 13.25V1.75C2 .784 2.784 0 3.75 0h5.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v8.586A1.75 1.75 0 0 1 12.25 15h-7a.75.75 0 0 1 0-1.5h7a.25.25 0 0 0 .25-.25V4.664a.25.25 0 0 0-.073-.177l-2.914-2.914a.25.25 0 0 0-.177-.073H3.75a.25.25 0 0 0-.25.25Z"/>
							</svg>
							<span class="prev-name">{sub.fileName}</span>
							<span class="prev-size">{formatSize(sub.fileSize)}</span>
						</div>
						<div class="prev-meta">
							<span class="prev-date">{formatDate(sub.createdAt)}</span>
							<span class="prev-status" class:reviewed={sub.status === 'reviewed'}>
								{sub.status}
							</span>
						</div>
					</div>

					<FileTreeBrowser submissionId={sub.id} fileName={sub.fileName} />

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
							{#if reviewingId === sub.id}
								<span class="spinner"></span>
								Reviewing...
							{:else}
								<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773.04-.074c.109-.187.303-.436.636-.663C7.246 4.28 7.6 4.125 8 4.125a2.137 2.137 0 0 1 1.268.425c.391.299.67.727.67 1.325 0 .35-.092.647-.257.9-.16.245-.373.41-.556.536-.073.05-.143.1-.207.143-.34.214-.418.315-.418.476a.75.75 0 0 1-1.5 0c0-.617.37-.965.755-1.21.128-.08.222-.145.297-.198.094-.067.145-.117.177-.166A.571.571 0 0 0 8.438 5.875c0-.212-.093-.35-.228-.454A.643.643 0 0 0 8 5.375a.626.626 0 0 0-.313.087.466.466 0 0 0-.15.161l-.02.035Z"/><path d="M9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"/></svg>
								Request AI Review
							{/if}
						</button>
					{/if}
				</div>
			{/each}
			{#if reviewError}
				<div class="error-msg">{reviewError}</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.submission-section {
		/* No border-top since it's inside a detail-group now */
	}

	/* Upload Zone */
	.upload-zone {
		position: relative;
		border: 2px dashed var(--border);
		border-radius: var(--radius-md, 8px);
		transition: all 0.2s ease;
		background: var(--bg-dark);
	}

	.upload-zone:hover,
	.upload-zone.drag-over {
		border-color: var(--purple);
		background: rgba(163, 113, 247, 0.05);
	}

	.upload-zone.has-file {
		border-color: var(--accent-bright);
		border-style: solid;
		background: rgba(63, 185, 80, 0.05);
	}

	.upload-label {
		display: block;
		cursor: pointer;
		padding: 16px;
	}

	.upload-empty {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		text-align: center;
	}

	.upload-icon {
		color: var(--text-muted);
		opacity: 0.5;
	}

	.upload-text {
		font-size: 13px;
		color: var(--text-secondary);
	}

	.upload-hint {
		font-size: 11px;
		color: var(--text-muted);
	}

	.upload-selected {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.upload-icon-selected {
		color: var(--accent-bright);
		flex-shrink: 0;
	}

	.selected-info {
		flex: 1;
		min-width: 0;
	}

	.selected-name {
		display: block;
		font-size: 13px;
		font-weight: 500;
		color: var(--text-primary);
		font-family: 'JetBrains Mono', monospace;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.selected-size {
		font-size: 11px;
		color: var(--text-muted);
	}

	.selected-check {
		color: var(--accent-bright);
		flex-shrink: 0;
	}

	.file-input {
		position: absolute;
		inset: 0;
		opacity: 0;
		cursor: pointer;
	}

	/* Form Actions */
	.form-actions {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-top: 12px;
	}

	.btn-submit {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		background: var(--accent);
		border: none;
		border-radius: 6px;
		color: white;
		cursor: pointer;
		font-size: 13px;
		font-weight: 500;
		font-family: inherit;
		transition: all 0.15s ease;
	}

	.btn-submit:hover:not(:disabled) {
		background: var(--accent-hover);
		box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,0.3));
	}

	.btn-submit:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-guide {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		background: none;
		border: 1px solid rgba(163, 113, 247, 0.3);
		border-radius: 6px;
		color: var(--purple);
		cursor: pointer;
		font-size: 13px;
		font-family: inherit;
		transition: all 0.15s ease;
	}

	.btn-guide:hover {
		background: rgba(163, 113, 247, 0.1);
		border-color: var(--purple);
	}

	.no-key-hint {
		font-size: 12px;
		color: var(--text-muted);
	}

	.no-key-hint a {
		color: var(--blue);
	}

	/* Error */
	.error-msg {
		padding: 8px 12px;
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.2);
		border-radius: 6px;
		color: var(--red);
		font-size: 13px;
		margin-top: 8px;
	}

	/* Spinner */
	.spinner {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 2px solid currentColor;
		border-top-color: transparent;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	/* Guide Section */
	.guide-section {
		margin-top: 14px;
		padding: 14px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 8px;
	}

	.guide-section textarea {
		width: 100%;
		padding: 10px 12px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-primary);
		font-family: inherit;
		font-size: 13px;
		resize: vertical;
		margin-bottom: 10px;
		transition: border-color 0.15s;
	}

	.guide-section textarea:focus {
		outline: none;
		border-color: var(--purple);
	}

	.btn-guide-send {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 6px 14px;
		background: var(--purple);
		border: none;
		border-radius: 6px;
		color: white;
		cursor: pointer;
		font-size: 13px;
		font-family: inherit;
		transition: opacity 0.15s;
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

	/* Previous Submissions */
	.previous-submissions {
		margin-top: 16px;
		padding-top: 16px;
		border-top: 1px solid var(--border);
	}

	.submissions-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 10px;
	}

	.prev-submission {
		margin-bottom: 10px;
		padding: 12px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 8px;
	}

	.prev-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		flex-wrap: wrap;
	}

	.prev-info {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}

	.prev-file-icon {
		color: var(--text-muted);
		flex-shrink: 0;
	}

	.prev-name {
		font-size: 12px;
		color: var(--text-primary);
		font-family: 'JetBrains Mono', monospace;
		font-weight: 500;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.prev-size {
		font-size: 11px;
		color: var(--text-muted);
		flex-shrink: 0;
	}

	.prev-meta {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-shrink: 0;
	}

	.prev-date {
		font-size: 11px;
		color: var(--text-muted);
	}

	.prev-status {
		padding: 2px 8px;
		background: rgba(210, 153, 34, 0.15);
		color: var(--orange);
		border-radius: 10px;
		font-size: 11px;
		text-transform: capitalize;
		font-weight: 500;
	}

	.prev-status.reviewed {
		background: rgba(63, 185, 80, 0.15);
		color: var(--accent-bright);
	}

	.btn-review {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin-top: 8px;
		padding: 5px 12px;
		background: rgba(163, 113, 247, 0.08);
		border: 1px solid rgba(163, 113, 247, 0.25);
		border-radius: 6px;
		color: var(--purple);
		cursor: pointer;
		font-size: 12px;
		font-family: inherit;
		transition: all 0.15s ease;
	}

	.btn-review:hover:not(:disabled) {
		background: rgba(163, 113, 247, 0.15);
		border-color: var(--purple);
	}

	.btn-review:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>

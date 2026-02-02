<script lang="ts">
	import { enhance } from '$app/forms';

	let { data, form } = $props();

	let showKey = $state(false);
	let saving = $state(false);
	let deleting = $state(false);
</script>

<svelte:head>
	<title>Settings - EduTutor Crafter</title>
</svelte:head>

<header class="page-header">
	<h1>Settings</h1>
	<p class="page-desc">Configure your EduTutor Crafter experience</p>
</header>

<div class="settings-content">
	<div class="settings-card">
		<div class="card-header">
			<div class="card-icon">🔑</div>
			<div>
				<h2>Gemini API Key</h2>
				<p class="card-desc">Required for AI code review and guidance features</p>
			</div>
		</div>

		<div class="card-body">
			{#if form?.success}
				<div class="alert success">{form.message}</div>
			{/if}
			{#if form?.error}
				<div class="alert error">{form.error}</div>
			{/if}

			<div class="status-row">
				<span class="status-label">Status:</span>
				{#if data.hasApiKey}
					<span class="status-badge configured">Configured</span>
					<span class="masked-key">{data.maskedApiKey}</span>
				{:else}
					<span class="status-badge not-configured">Not configured</span>
				{/if}
			</div>

			<form
				method="POST"
				action="?/saveApiKey"
				use:enhance={() => {
					saving = true;
					return async ({ update }) => {
						saving = false;
						await update();
					};
				}}
			>
				<div class="form-group">
					<label for="apiKey">API Key</label>
					<div class="input-wrapper">
						{#if showKey}
							<input
								type="text"
								id="apiKey"
								name="apiKey"
								placeholder="Enter your Gemini API key"
								autocomplete="off"
							/>
						{:else}
							<input
								type="password"
								id="apiKey"
								name="apiKey"
								placeholder="Enter your Gemini API key"
								autocomplete="off"
							/>
						{/if}
						<button type="button" class="toggle-btn" onclick={() => (showKey = !showKey)}>
							{showKey ? 'Hide' : 'Show'}
						</button>
					</div>
					<p class="form-hint">
						Get your API key from <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">Google AI Studio</a>
					</p>
				</div>
				<button type="submit" class="btn primary" disabled={saving}>
					{saving ? 'Saving...' : 'Save API Key'}
				</button>
			</form>

			{#if data.hasApiKey}
				<form
					method="POST"
					action="?/deleteApiKey"
					use:enhance={() => {
						deleting = true;
						return async ({ update }) => {
							deleting = false;
							await update();
						};
					}}
				>
					<button type="submit" class="btn danger" disabled={deleting}>
						{deleting ? 'Deleting...' : 'Delete API Key'}
					</button>
				</form>
			{/if}
		</div>
	</div>
</div>

<style>
	.page-header {
		padding: 24px 32px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
	}

	.page-header h1 {
		font-size: 24px;
		font-weight: 700;
	}

	.page-desc {
		color: var(--text-secondary);
		font-size: 14px;
		margin-top: 4px;
	}

	.settings-content {
		padding: 24px 32px;
		max-width: 640px;
	}

	.settings-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
	}

	.card-header {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 16px 20px;
		border-bottom: 1px solid var(--border);
	}

	.card-icon {
		font-size: 24px;
	}

	.card-header h2 {
		font-size: 16px;
		font-weight: 600;
	}

	.card-desc {
		font-size: 13px;
		color: var(--text-muted);
	}

	.card-body {
		padding: 20px;
	}

	.alert {
		padding: 10px 14px;
		border-radius: 6px;
		font-size: 13px;
		margin-bottom: 16px;
	}

	.alert.success {
		background: rgba(63, 185, 80, 0.1);
		border: 1px solid rgba(63, 185, 80, 0.3);
		color: var(--accent-bright);
	}

	.alert.error {
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.3);
		color: var(--red);
	}

	.status-row {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 20px;
	}

	.status-label {
		font-size: 13px;
		color: var(--text-muted);
	}

	.status-badge {
		padding: 2px 8px;
		border-radius: 10px;
		font-size: 12px;
		font-weight: 500;
	}

	.status-badge.configured {
		background: rgba(63, 185, 80, 0.15);
		color: var(--accent-bright);
	}

	.status-badge.not-configured {
		background: rgba(210, 153, 34, 0.15);
		color: var(--orange);
	}

	.masked-key {
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		color: var(--text-muted);
	}

	.form-group {
		margin-bottom: 16px;
	}

	.form-group label {
		display: block;
		font-size: 13px;
		font-weight: 500;
		margin-bottom: 6px;
		color: var(--text-secondary);
	}

	.input-wrapper {
		display: flex;
		gap: 8px;
	}

	.input-wrapper input {
		flex: 1;
		padding: 8px 12px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-primary);
		font-family: 'JetBrains Mono', monospace;
		font-size: 13px;
	}

	.input-wrapper input:focus {
		outline: none;
		border-color: var(--blue);
	}

	.toggle-btn {
		padding: 8px 12px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-muted);
		cursor: pointer;
		font-size: 12px;
		font-family: inherit;
	}

	.toggle-btn:hover {
		color: var(--text-primary);
		border-color: var(--text-muted);
	}

	.form-hint {
		font-size: 12px;
		color: var(--text-muted);
		margin-top: 6px;
	}

	.form-hint a {
		color: var(--blue);
	}

	.btn {
		padding: 8px 16px;
		border: none;
		border-radius: 6px;
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		font-family: inherit;
		transition: opacity 0.15s;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn.primary {
		background: var(--accent);
		color: white;
	}

	.btn.primary:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	.btn.danger {
		background: none;
		border: 1px solid var(--red);
		color: var(--red);
		margin-top: 12px;
	}

	.btn.danger:hover:not(:disabled) {
		background: rgba(248, 81, 73, 0.1);
	}

	@media (max-width: 768px) {
		.page-header {
			padding: 16px;
		}
		.settings-content {
			padding: 16px;
		}
	}
</style>

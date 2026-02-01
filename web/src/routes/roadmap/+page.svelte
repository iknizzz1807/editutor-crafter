<script lang="ts">
	let { data } = $props();
</script>

<svelte:head>
	<title>Roadmap - EduTutor Crafter</title>
</svelte:head>

<header class="main-header">
	<h1>📚 Learning Roadmap</h1>
	<p>Choose a domain to start your learning journey</p>
</header>

<div class="content">
	<div class="domain-grid">
		{#each data.domains as domain}
			<a href="/roadmap/{domain.slug}" class="domain-card">
				<div class="domain-card-icon">{domain.icon || '📁'}</div>
				<div class="domain-card-info">
					<h2>{domain.name}</h2>
					<p>{domain.description || ''}</p>
				</div>
				<div class="domain-card-stats">
					<span class="stat">{domain.totalMilestones} milestones</span>
					{#if domain.completedMilestones > 0}
						<span class="stat completed"
							>{domain.completedMilestones}/{domain.totalMilestones} done</span
						>
					{/if}
				</div>
			</a>
		{/each}
	</div>
</div>

<style>
	.main-header {
		padding: 24px 32px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
		position: sticky;
		top: 0;
		z-index: 50;
	}

	.main-header h1 {
		font-size: 24px;
		font-weight: 700;
	}

	.main-header p {
		color: var(--text-secondary);
		margin-top: 4px;
		font-size: 14px;
	}

	.content {
		padding: 24px 32px;
	}

	.domain-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
		gap: 16px;
	}

	.domain-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 20px;
		text-decoration: none;
		color: var(--text-primary);
		transition: all 0.2s ease;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.domain-card:hover {
		border-color: var(--text-muted);
		text-decoration: none;
	}

	.domain-card-icon {
		font-size: 32px;
	}

	.domain-card-info h2 {
		font-size: 16px;
		font-weight: 600;
		margin-bottom: 4px;
	}

	.domain-card-info p {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.domain-card-stats {
		display: flex;
		gap: 12px;
	}

	.stat {
		font-size: 12px;
		color: var(--text-muted);
		padding: 4px 8px;
		background: var(--bg-dark);
		border-radius: 4px;
	}

	.stat.completed {
		color: var(--accent-bright);
		background: rgba(63, 185, 80, 0.1);
	}

	@media (max-width: 768px) {
		.content {
			padding: 16px;
		}

		.domain-grid {
			grid-template-columns: 1fr;
		}
	}
</style>

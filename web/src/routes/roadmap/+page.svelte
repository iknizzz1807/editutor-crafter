<script lang="ts">
	import DomainIcon from '$lib/components/DomainIcon.svelte';
	let { data } = $props();
</script>

<svelte:head>
	<title>Roadmap - EduTutor Crafter</title>
</svelte:head>

<header class="main-header">
	<h1>
		<svg width="24" height="24" viewBox="0 0 16 16" fill="var(--blue)">
			<path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm7.251 10.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574zM8.755 4.75l-.004 7.322a3.752 3.752 0 011.992-.572H14.5v-9h-3.495a2.25 2.25 0 00-2.25 2.25z"/>
		</svg>
		Learning Roadmap
	</h1>
	<p>Choose a domain to start your learning journey</p>
</header>

<div class="content">
	<div class="domain-grid">
		{#each data.domains as domain}
			{@const totalM = domain.totalMilestones || 0}
			{@const completedM = domain.completedMilestones || 0}
			{@const pct = totalM > 0 ? (completedM / totalM) * 100 : 0}
			<a href="/roadmap/{domain.slug}" class="domain-card">
				<div class="domain-card-top">
					<div class="domain-card-icon">
					<DomainIcon icon={domain.icon} />
				</div>
					<div class="domain-card-info">
						<h2>{domain.name}</h2>
						{#if domain.description}
							<p>{domain.description}</p>
						{/if}
					</div>
				</div>

				<div class="domain-card-bottom">
					<div class="domain-stats">
						<span class="stat-label">{totalM} milestones</span>
						{#if completedM > 0}
							<span class="stat-done">{completedM} done</span>
						{/if}
						{#if pct > 0}
							<span class="stat-pct">{Math.round(pct)}%</span>
						{/if}
					</div>
					{#if totalM > 0}
						<div class="domain-progress">
							<div
								class="domain-progress-fill"
								class:complete={pct === 100}
								style="width: {pct}%"
							></div>
						</div>
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
		display: flex;
		align-items: center;
		gap: 10px;
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
		grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
		gap: 16px;
	}

	.domain-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 20px;
		text-decoration: none;
		color: var(--text-primary);
		transition: all 0.2s ease;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.domain-card:hover {
		border-color: var(--text-muted);
		box-shadow: var(--shadow-md);
		text-decoration: none;
		transform: translateY(-2px);
	}

	.domain-card-top {
		display: flex;
		gap: 14px;
		align-items: flex-start;
	}

	.domain-card-icon {
		font-size: 32px;
		width: 48px;
		height: 48px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--bg-elevated);
		border-radius: var(--radius-md);
		flex-shrink: 0;
	}

	.domain-card-info {
		flex: 1;
		min-width: 0;
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
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	.domain-card-bottom {
		margin-top: auto;
	}

	.domain-stats {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 8px;
	}

	.stat-label {
		font-size: 12px;
		color: var(--text-muted);
	}

	.stat-done {
		font-size: 12px;
		color: var(--accent-bright);
	}

	.stat-pct {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-secondary);
		margin-left: auto;
	}

	.domain-progress {
		height: 6px;
		background: var(--bg-dark);
		border-radius: 3px;
		overflow: hidden;
	}

	.domain-progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-bright), var(--blue));
		border-radius: 3px;
		transition: width 0.5s ease;
	}

	.domain-progress-fill.complete {
		background: var(--accent-bright);
	}

	@media (max-width: 768px) {
		.main-header {
			padding: 16px;
		}

		.content {
			padding: 16px;
		}

		.domain-grid {
			grid-template-columns: 1fr;
		}
	}
</style>

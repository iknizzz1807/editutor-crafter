<script lang="ts">
	import { enhance } from '$app/forms';

	let {
		mode,
		error,
		username,
		email
	}: {
		mode: 'login' | 'register';
		error?: string;
		username?: string;
		email?: string;
	} = $props();

	let isLogin = $derived(mode === 'login');
</script>

<div class="auth-container">
	<div class="auth-card">
		<div class="auth-header">
			<div class="logo">
				<span class="logo-icon">🎯</span>
				EduTutor Crafter
			</div>
			<h1>{isLogin ? 'Sign In' : 'Create Account'}</h1>
			<p>{isLogin ? 'Welcome back! Sign in to continue your learning journey.' : 'Start your learning journey today.'}</p>
		</div>

		{#if error}
			<div class="error-message">{error}</div>
		{/if}

		<form method="POST" use:enhance>
			<div class="form-group">
				<label for="username">Username</label>
				<input
					type="text"
					id="username"
					name="username"
					value={username ?? ''}
					required
					autocomplete="username"
				/>
			</div>

			{#if !isLogin}
				<div class="form-group">
					<label for="email">Email</label>
					<input
						type="email"
						id="email"
						name="email"
						value={email ?? ''}
						required
						autocomplete="email"
					/>
				</div>
			{/if}

			<div class="form-group">
				<label for="password">Password</label>
				<input
					type="password"
					id="password"
					name="password"
					required
					autocomplete={isLogin ? 'current-password' : 'new-password'}
				/>
			</div>

			{#if !isLogin}
				<div class="form-group">
					<label for="confirmPassword">Confirm Password</label>
					<input
						type="password"
						id="confirmPassword"
						name="confirmPassword"
						required
						autocomplete="new-password"
					/>
				</div>
			{/if}

			<button type="submit" class="submit-btn">
				{isLogin ? 'Sign In' : 'Create Account'}
			</button>
		</form>

		<div class="auth-footer">
			{#if isLogin}
				<p>Don't have an account? <a href="/auth/register">Register</a></p>
			{:else}
				<p>Already have an account? <a href="/auth/login">Sign In</a></p>
			{/if}
		</div>
	</div>
</div>

<style>
	.auth-container {
		min-height: 100vh;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 20px;
	}

	.auth-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 12px;
		padding: 40px;
		width: 100%;
		max-width: 400px;
	}

	.auth-header {
		text-align: center;
		margin-bottom: 32px;
	}

	.logo {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		font-weight: 700;
		font-size: 18px;
		margin-bottom: 24px;
	}

	.logo-icon {
		width: 32px;
		height: 32px;
		background: linear-gradient(135deg, var(--accent-bright), var(--blue));
		border-radius: 8px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 16px;
	}

	h1 {
		font-size: 24px;
		font-weight: 700;
		margin-bottom: 8px;
	}

	.auth-header p {
		color: var(--text-secondary);
		font-size: 14px;
	}

	.error-message {
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.3);
		color: var(--red);
		padding: 10px 14px;
		border-radius: 6px;
		font-size: 13px;
		margin-bottom: 20px;
	}

	.form-group {
		margin-bottom: 16px;
	}

	label {
		display: block;
		font-size: 13px;
		font-weight: 500;
		color: var(--text-secondary);
		margin-bottom: 6px;
	}

	input {
		width: 100%;
		padding: 10px 14px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-primary);
		font-size: 14px;
		font-family: inherit;
		transition: border-color 0.15s ease;
	}

	input:focus {
		outline: none;
		border-color: var(--accent-bright);
	}

	.submit-btn {
		width: 100%;
		padding: 12px;
		background: var(--accent);
		border: none;
		border-radius: 6px;
		color: white;
		font-size: 14px;
		font-weight: 600;
		font-family: inherit;
		cursor: pointer;
		transition: background 0.15s ease;
		margin-top: 8px;
	}

	.submit-btn:hover {
		background: var(--accent-hover);
	}

	.auth-footer {
		text-align: center;
		margin-top: 24px;
		font-size: 13px;
		color: var(--text-secondary);
	}

	.auth-footer a {
		color: var(--blue);
		text-decoration: none;
	}

	.auth-footer a:hover {
		text-decoration: underline;
	}
</style>

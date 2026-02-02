import { writable } from 'svelte/store';
import { browser } from '$app/environment';

const STORAGE_KEY = 'sidebar-collapsed';

const initial = browser ? localStorage.getItem(STORAGE_KEY) === 'true' : false;

export const sidebarCollapsed = writable<boolean>(initial);

if (browser) {
	sidebarCollapsed.subscribe((value) => {
		localStorage.setItem(STORAGE_KEY, String(value));
	});
}

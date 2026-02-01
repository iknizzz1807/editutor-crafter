import { db } from './db/index.js';
import { users, sessions } from './db/schema.js';
import { eq, and, gt } from 'drizzle-orm';
import bcrypt from 'bcrypt';
import crypto from 'crypto';

const SALT_ROUNDS = 10;
const SESSION_DURATION_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

export async function hashPassword(password: string): Promise<string> {
	return bcrypt.hash(password, SALT_ROUNDS);
}

export async function verifyPassword(password: string, hash: string): Promise<boolean> {
	return bcrypt.compare(password, hash);
}

export function generateSessionToken(): string {
	return crypto.randomBytes(32).toString('hex');
}

export function createSession(userId: number): { token: string; expiresAt: string } {
	const token = generateSessionToken();
	const expiresAt = new Date(Date.now() + SESSION_DURATION_MS).toISOString();

	db.insert(sessions).values({ userId, token, expiresAt }).run();

	return { token, expiresAt };
}

export function validateSession(token: string): { id: number; username: string; email: string } | null {
	const now = new Date().toISOString();

	const result = db
		.select({
			userId: sessions.userId,
			username: users.username,
			email: users.email,
			expiresAt: sessions.expiresAt
		})
		.from(sessions)
		.innerJoin(users, eq(sessions.userId, users.id))
		.where(and(eq(sessions.token, token), gt(sessions.expiresAt, now)))
		.get();

	if (!result) return null;

	return { id: result.userId, username: result.username, email: result.email };
}

export function deleteSession(token: string): void {
	db.delete(sessions).where(eq(sessions.token, token)).run();
}

export function createUser(
	username: string,
	email: string,
	passwordHash: string
): { id: number } {
	const result = db
		.insert(users)
		.values({ username, email, passwordHash })
		.returning({ id: users.id })
		.get();
	return result;
}

export function findUserByUsername(username: string) {
	return db.select().from(users).where(eq(users.username, username)).get();
}

export function findUserByEmail(email: string) {
	return db.select().from(users).where(eq(users.email, email)).get();
}

import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const KEY_LENGTH = 32;
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

function getKey(): Buffer {
	const secret = process.env.ENCRYPTION_KEY || 'editutor-crafter-default-secret-key';
	return scryptSync(secret, 'editutor-salt', KEY_LENGTH);
}

export function encrypt(text: string): string {
	const key = getKey();
	const iv = randomBytes(IV_LENGTH);
	const cipher = createCipheriv(ALGORITHM, key, iv);

	let encrypted = cipher.update(text, 'utf8', 'hex');
	encrypted += cipher.final('hex');
	const authTag = cipher.getAuthTag();

	// Format: iv:authTag:encrypted
	return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
}

export function decrypt(ciphertext: string): string {
	const key = getKey();
	const [ivHex, authTagHex, encrypted] = ciphertext.split(':');

	const iv = Buffer.from(ivHex, 'hex');
	const authTag = Buffer.from(authTagHex, 'hex');
	const decipher = createDecipheriv(ALGORITHM, key, iv);
	decipher.setAuthTag(authTag);

	let decrypted = decipher.update(encrypted, 'hex', 'utf8');
	decrypted += decipher.final('utf8');
	return decrypted;
}

export function maskApiKey(key: string): string {
	if (key.length <= 8) return '****';
	return key.slice(0, 4) + '****' + key.slice(-4);
}

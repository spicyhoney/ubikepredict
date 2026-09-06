const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
const allowLocalBuild = process.env.ALLOW_LOCAL_API_BUILD === 'true';

let parsedUrl;
if (apiBaseUrl) {
  try {
    parsedUrl = new URL(apiBaseUrl);
  } catch {
    console.error('[build] NEXT_PUBLIC_API_BASE_URL must be a valid absolute URL.');
    process.exit(1);
  }
}

const isLoopback = parsedUrl
  ? ['localhost', '127.0.0.1', '::1'].includes(parsedUrl.hostname)
  : true;
const isSafePublicUrl = parsedUrl?.protocol === 'https:' && !isLoopback;
const isIntentionalLocalUrl = parsedUrl
  ? ['http:', 'https:'].includes(parsedUrl.protocol) && isLoopback
  : true;

if (!isSafePublicUrl && !(allowLocalBuild && isIntentionalLocalUrl)) {
  console.error(
    '[build] NEXT_PUBLIC_API_BASE_URL must be an absolute, public HTTPS URL. ' +
      'Set ALLOW_LOCAL_API_BUILD=true only when intentionally previewing the production bundle locally.',
  );
  process.exit(1);
}

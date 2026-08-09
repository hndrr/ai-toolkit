// middleware.ts (at the root of your project)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Browser media elements cannot attach an Authorization header. The auth
// endpoint sets this HttpOnly session cookie after validating the password.
export const AUTH_COOKIE_NAME = 'AI_TOOLKIT_SESSION';

function requestToken(request: NextRequest): string | null {
  const authorization = request.headers.get('Authorization');
  if (authorization !== null) {
    const match = authorization.match(/^Bearer\s+(.+)$/i);
    return match ? match[1] : '';
  }
  return request.cookies.get(AUTH_COOKIE_NAME)?.value ?? null;
}

export function middleware(request: NextRequest) {
  // If AI_TOOLKIT_AUTH is not configured, preserve the local unauthenticated
  // development behavior. Beam/Modal deployments provide it through a secret.
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) {
    return NextResponse.next();
  }

  const token = requestToken(request);
  if (token !== tokenToUse) {
    const response = NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    // An invalid Authorization header must not fall back to an old session.
    response.cookies.delete(AUTH_COOKIE_NAME);
    return response;
  }

  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};

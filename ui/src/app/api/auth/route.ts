import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { AUTH_COOKIE_NAME } from '@/middleware';

export async function GET(request: NextRequest) {
  // if this gets hit, auth has already been verified
  const response = NextResponse.json({ isAuthenticated: true });
  if (process.env.AI_TOOLKIT_AUTH) {
    const forwardedProtocol = request.headers.get('x-forwarded-proto')?.split(',')[0].trim();
    const isSecureRequest = forwardedProtocol
      ? forwardedProtocol === 'https'
      : request.nextUrl.protocol === 'https:';
    response.cookies.set(AUTH_COOKIE_NAME, process.env.AI_TOOLKIT_AUTH, {
      httpOnly: true,
      sameSite: 'lax',
      secure: isSecureRequest,
      path: '/',
    });
  }
  return response;
}

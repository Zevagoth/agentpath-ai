/**
 * Give the jsdom environment a working `fetch`.
 *
 * jsdom implements XHR but not fetch, and Vitest's jsdom environment does not
 * hand Node's implementation through, so `fetch` is missing inside these tests.
 * Without this the console component reports "cannot reach the API" and the
 * failure looks like an outage rather than a missing global.
 *
 * undici is Node's own HTTP client, so this is the same stack the browser-side
 * fetch talks to in production, minus CORS enforcement (which the API allows for
 * loopback origins anyway).
 */

import { FormData, Headers, Request, Response, fetch } from "undici";

Object.assign(globalThis, { fetch, Headers, Request, Response, FormData });

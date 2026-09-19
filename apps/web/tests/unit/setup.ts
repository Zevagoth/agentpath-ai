import "@testing-library/dom";

/**
 * A `Storage` for the test environment.
 *
 * This jsdom build does not install `window.localStorage`, so the property is
 * simply absent and Node's own experimental global is what a lookup would
 * otherwise find. The feedback store treats a missing storage as "do not
 * persist", which is the right production behaviour and means its persistence
 * would silently go untested - the assertions would pass against a no-op.
 *
 * So a conforming in-memory implementation is installed, and only when the
 * environment has not provided a real one. The module under test is unchanged and
 * its no-storage path still has its own test.
 */
class MemoryStorage implements Storage {
  private entries = new Map<string, string>();

  get length(): number {
    return this.entries.size;
  }

  clear(): void {
    this.entries.clear();
  }

  getItem(key: string): string | null {
    return this.entries.has(key) ? (this.entries.get(key) as string) : null;
  }

  key(index: number): string | null {
    return [...this.entries.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.entries.delete(key);
  }

  setItem(key: string, value: string): void {
    this.entries.set(key, String(value));
  }
}

if (typeof window !== "undefined" && !window.localStorage) {
  Object.defineProperty(window, "localStorage", {
    value: new MemoryStorage(),
    configurable: true,
    writable: false,
  });
}

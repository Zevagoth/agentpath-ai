"use client";

import type {
  BookingStatus,
  CaseContextInput,
  PaymentStatus,
  SupplierStatus,
} from "@/lib/types";

/**
 * The facts the agent already has, entered by hand.
 *
 * Every field here mirrors `agentpath.case_context.CaseContext`, and the fields
 * that are *missing* are the point of the panel. There is no name, no booking
 * reference, no phone number and no card detail, because the API declares
 * `extra="forbid"`: sending one is rejected with HTTP 422 rather than quietly
 * ignored, so the field never exists to be logged or put in a prompt. The note at
 * the bottom says so on screen, since a privacy property nobody can see is
 * indistinguishable from a claim.
 *
 * Collapsed by default. An agent describing a case should not have to fill in a
 * form first; this is here for when the extra facts change the branch.
 */

const BOOKING_STATUSES: BookingStatus[] = [
  "confirmed",
  "pending",
  "cancelled",
  "modified",
  "no_show",
  "completed",
];

const SUPPLIER_STATUSES: SupplierStatus[] = [
  "not_contacted",
  "unreachable",
  "confirmed_available",
  "cannot_accommodate",
  "no_record_of_booking",
  "disputed",
];

const PAYMENT_STATUSES: PaymentStatus[] = [
  "not_paid",
  "prepaid",
  "pay_at_property",
  "refund_requested",
  "refund_issued",
  "chargeback",
];

/** A worked example, so the panel can be demonstrated in one click. */
export const WALKED_GUEST_EXAMPLE: CaseContextInput = {
  booking_status: "confirmed",
  supplier_status: "cannot_accommodate",
  payment_status: "prepaid",
  customer_arrived: true,
  party_size: 2,
  rooms_booked: 1,
  service_date: "2026-04-16",
  market: "JO",
};

export function isEmptyContext(context: CaseContextInput): boolean {
  return Object.values(context).every(
    (value) => value === null || value === undefined || value === "" || (Array.isArray(value) && value.length === 0),
  );
}

export function CaseContextForm({
  value,
  onChange,
  disabled,
}: {
  value: CaseContextInput;
  onChange: (next: CaseContextInput) => void;
  disabled?: boolean;
}) {
  const set = <K extends keyof CaseContextInput>(key: K, next: CaseContextInput[K]) =>
    onChange({ ...value, [key]: next });

  const count = Object.values(value).filter(
    (item) => item !== null && item !== undefined && item !== "" && !(Array.isArray(item) && item.length === 0),
  ).length;

  return (
    <details className="context-panel" data-testid="case-context-form">
      <summary className="context-summary">
        <span>Case facts (optional, synthetic)</span>
        <span className="context-count" data-testid="context-field-count">
          {count === 0 ? "none set" : `${count} set`}
        </span>
      </summary>

      <div className="context-grid">
        <Field label="Booking status" htmlFor="ctx-booking">
          <select
            id="ctx-booking"
            className="select select-compact"
            data-testid="ctx-booking-status"
            disabled={disabled}
            value={value.booking_status ?? ""}
            onChange={(event) => set("booking_status", (event.target.value || null) as BookingStatus | null)}
          >
            <option value="">unknown</option>
            {BOOKING_STATUSES.map((status) => (
              <option key={status} value={status}>
                {status.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Supplier said" htmlFor="ctx-supplier">
          <select
            id="ctx-supplier"
            className="select select-compact"
            data-testid="ctx-supplier-status"
            disabled={disabled}
            value={value.supplier_status ?? ""}
            onChange={(event) => set("supplier_status", (event.target.value || null) as SupplierStatus | null)}
          >
            <option value="">unknown</option>
            {SUPPLIER_STATUSES.map((status) => (
              <option key={status} value={status}>
                {status.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Payment stage" htmlFor="ctx-payment">
          <select
            id="ctx-payment"
            className="select select-compact"
            data-testid="ctx-payment-status"
            disabled={disabled}
            value={value.payment_status ?? ""}
            onChange={(event) => set("payment_status", (event.target.value || null) as PaymentStatus | null)}
          >
            <option value="">unknown</option>
            {PAYMENT_STATUSES.map((status) => (
              <option key={status} value={status}>
                {status.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Customer is present" htmlFor="ctx-arrived">
          <select
            id="ctx-arrived"
            className="select select-compact"
            data-testid="ctx-customer-arrived"
            disabled={disabled}
            value={value.customer_arrived === null || value.customer_arrived === undefined ? "" : String(value.customer_arrived)}
            onChange={(event) =>
              set("customer_arrived", event.target.value === "" ? null : event.target.value === "true")
            }
          >
            <option value="">unknown</option>
            <option value="true">yes, at the property now</option>
            <option value="false">no</option>
          </select>
        </Field>

        <Field label="Guests" htmlFor="ctx-party">
          <input
            id="ctx-party"
            className="select select-compact"
            data-testid="ctx-party-size"
            type="number"
            min={1}
            max={99}
            disabled={disabled}
            value={value.party_size ?? ""}
            onChange={(event) => set("party_size", toCount(event.target.value))}
          />
        </Field>

        <Field label="Rooms" htmlFor="ctx-rooms">
          <input
            id="ctx-rooms"
            className="select select-compact"
            data-testid="ctx-rooms-booked"
            type="number"
            min={1}
            max={99}
            disabled={disabled}
            value={value.rooms_booked ?? ""}
            onChange={(event) => set("rooms_booked", toCount(event.target.value))}
          />
        </Field>

        <Field label="Service date" htmlFor="ctx-date">
          <input
            id="ctx-date"
            className="select select-compact"
            data-testid="ctx-service-date"
            type="date"
            disabled={disabled}
            value={value.service_date ?? ""}
            onChange={(event) => set("service_date", event.target.value || null)}
          />
        </Field>

        <Field label="Market" htmlFor="ctx-market">
          <input
            id="ctx-market"
            className="select select-compact"
            data-testid="ctx-market"
            placeholder="JO"
            maxLength={8}
            disabled={disabled}
            value={value.market ?? ""}
            onChange={(event) => set("market", event.target.value || null)}
          />
        </Field>

        <Field label="Answer language" htmlFor="ctx-language">
          <select
            id="ctx-language"
            className="select select-compact"
            data-testid="ctx-language"
            disabled={disabled}
            value={value.language ?? ""}
            onChange={(event) => set("language", (event.target.value || null) as "ar" | "en" | null)}
          >
            <option value="">detect from the case</option>
            <option value="ar">العربية</option>
            <option value="en">English</option>
          </select>
        </Field>

        <Field label="Console error code" htmlFor="ctx-error">
          <input
            id="ctx-error"
            className="select select-compact"
            data-testid="ctx-error-code"
            placeholder="SUPPLIER_TIMEOUT"
            maxLength={40}
            disabled={disabled}
            value={value.error_code ?? ""}
            onChange={(event) => set("error_code", event.target.value || null)}
          />
        </Field>

        <Field label="Steps already done" htmlFor="ctx-done" wide>
          <input
            id="ctx-done"
            className="select select-compact"
            data-testid="ctx-actions-taken"
            placeholder="NW-HTL-014#S01, NW-HTL-014#S02"
            disabled={disabled}
            value={(value.actions_already_taken ?? []).join(", ")}
            onChange={(event) =>
              set(
                "actions_already_taken",
                event.target.value
                  .split(",")
                  .map((token) => token.trim())
                  .filter(Boolean),
              )
            }
          />
        </Field>
      </div>

      <div className="context-actions">
        <button
          type="button"
          className="example-chip"
          data-testid="ctx-load-example"
          disabled={disabled}
          onClick={() => onChange({ ...WALKED_GUEST_EXAMPLE })}
        >
          Load the walked-guest example
        </button>
        <button
          type="button"
          className="example-chip"
          data-testid="ctx-clear"
          disabled={disabled}
          onClick={() => onChange({})}
        >
          Clear
        </button>
      </div>

      <p className="context-privacy" data-testid="context-privacy-note">
        There is no field here for a guest name, booking reference, phone number or
        card detail, and that is enforced rather than promised: the API rejects those
        keys outright, so they cannot be logged, cached or put in a prompt. Because
        we will not hold them, the assistant also never asks for them. Every value
        above is fabricated demo data.
      </p>
    </details>
  );
}

function Field({
  label,
  htmlFor,
  wide,
  children,
}: {
  label: string;
  htmlFor: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={wide ? "context-field context-field-wide" : "context-field"}>
      <label className="context-label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
    </div>
  );
}

function toCount(raw: string): number | null {
  if (raw.trim() === "") return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

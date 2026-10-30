---
doc_id: INC-OPS-005
title: Phishing campaign targeting branch staff
department: security
document_type: incident
access_level: internal
created_date: '2026-05-13'
source_uri: docs://security/INC-OPS-005
---

# Phishing campaign targeting branch staff

## Summary

This report covers an incident in the security department: Phishing campaign targeting branch staff. It records the cause, the response and the improvements that were agreed.

## Root cause

Staff received emails imitating the payroll portal, and two accounts were locked as a precaution. The issue was found through routine monitoring and was contained without any effect on customer payments.

## Resolution

The owning team corrected the configuration and confirmed normal behaviour. The incident lead reviewed the timeline with everyone involved.

## Follow-up actions

The related runbook was updated, and the alert thresholds were reviewed so that the same condition is caught earlier next time.

## Contributing factors

The post-incident review looked beyond the immediate cause of the the security service incident. Monitoring covered the symptoms, but it did not give an early warning. The on-call engineer had to piece together information from several dashboards, which slowed the diagnosis. Ownership of the affected component had changed in the last year, and the handover notes did not mention this risk. None of these factors caused the incident alone, but together they increased the time customers were affected.

## What went well

The alert reached the on-call engineer quickly, and the incident bridge was open within minutes. Clear roles helped: the incident lead handled communication while engineers focused on the security service. Status updates kept the contact centre informed, so advisers could give customers accurate information. The runbook helped the team confirm recovery, and reconciliation after the incident found no lasting financial differences.

## Lessons learned

The review concluded that the team should detect this kind of failure before customers do. Each lesson from the the security service incident was turned into a tracked action rather than a general intention. The team will share this report with other service owners, because similar risks may exist elsewhere. The report will be revisited at the next reliability review, to confirm that the actions have actually reduced the risk.

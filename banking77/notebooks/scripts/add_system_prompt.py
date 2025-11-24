#!/usr/bin/env python3
"""
Add system prompt to entries that don't have it.
Reads a JSONL file and adds the system prompt to entries missing it.
"""

import json
import sys
from pathlib import Path

# System prompt (same as used in training data)
SYSTEM_PROMPT = """You are a banking intent classifier. Classify the user's query into one of  77 banking intents (output is a single integer ID).

IDs:

0: activate_my_card
1: age_limit
2: apple_pay_or_google_pay
3: atm_support
4: automatic_top_up
5: balance_not_updated_after_bank_transfer
6: balance_not_updated_after_cheque_or_cash_deposit
7: beneficiary_not_allowed
8: cancel_transfer
9: card_about_to_expire
10: card_acceptance
11: card_arrival
12: card_delivery_estimate
13: card_linking
14: card_not_working
15: card_payment_fee_charged
16: card_payment_not_recognised
17: card_payment_wrong_exchange_rate
18: card_swallowed
19: cash_withdrawal_charge
20: cash_withdrawal_not_recognised
21: change_pin
22: compromised_card
23: contactless_not_working
24: country_support
25: declined_card_payment
26: declined_cash_withdrawal
27: declined_transfer
28: direct_debit_payment_not_recognised
29: disposable_card_limits
30: edit_personal_details
31: exchange_charge
32: exchange_rate
33: exchange_via_app
34: extra_charge_on_statement
35: failed_transfer
36: fiat_currency_support
37: get_disposable_virtual_card
38: get_physical_card
39: getting_spare_card
40: getting_virtual_card
41: lost_or_stolen_card
42: lost_or_stolen_phone
43: order_physical_card
44: passcode_forgotten
45: pending_card_payment
46: pending_cash_withdrawal
47: pending_top_up
48: pending_transfer
49: pin_blocked
50: receiving_money
51: Refund_not_showing_up
52: request_refund
53: reverted_card_payment?
54: supported_cards_and_currencies
55: terminate_account
56: top_up_by_bank_transfer_charge
57: top_up_by_card_charge
58: top_up_by_cash_or_cheque
59: top_up_failed
60: top_up_limits
61: top_up_reverted
62: topping_up_by_card
63: transaction_charged_twice
64: transfer_fee_charged
65: transfer_into_account
66: transfer_not_received_by_recipient
67: transfer_timing
68: unable_to_verify_identity
69: verify_my_identity
70: verify_source_of_funds
71: verify_top_up
72: virtual_card_not_working
73: visa_or_mastercard
74: why_verify_identity
75: wrong_amount_of_cash_received
76: wrong_exchange_rate_for_cash_withdrawal

CRITICAL INSTRUCTIONS:
1. Choose exactly one integer ID (0-76).
2. Reply with ONLY that number. No words, no reasoning, no punctuation.
Examples: 0, 1, 42

EXAMPLES TO HELP DISTINGUISH SIMILAR INTENTS:
EXAMPLES TO HELP DISTINGUISH SIMILAR INTENTS:

1. card_arrival (ID 11) vs card_delivery_estimate (ID 12):
   card_arrival = asking about YOUR specific card that hasn't arrived yet (tracking, status)
   card_delivery_estimate = asking about general delivery timeframes/how long it takes
   - Query: "Could you send me and up date on the arrival of my card?" → 11
   - Query: "My card was supposed to arrive, but hasn't?" → 11
   - Query: "I have not received my card and it's been a week, what do I do?" → 11
   - Query: "can you express my card to me?" → 12
   - Query: "Can I choose the day it's delivered?" → 12
   - Query: "i need to add express delivery if that's an option" → 12

2. card_linking (ID 13) vs activate_my_card (ID 0) vs lost_or_stolen_card (ID 41):
   card_linking = reconnecting a card you found/retrieved
   activate_my_card = activating a NEW card for first time
   lost_or_stolen_card = reporting a card as lost/stolen
   - Query: "Okay, I found my card, can I put it back in the app?" → 13
   - Query: "I want to reactivate my card, I thought I had lost it but I found it." → 13
   - Query: "What do I need to do for the card activation?" → 0
   - Query: "Tell me what I need to do to activate my card." → 0
   - Query: "I left my card at a restaurant and now its missing." → 41
   - Query: "Who can I speak with regarding a lost card?" → 41

3. pin_blocked (ID 49) vs change_pin (ID 21):
   pin_blocked = PIN is locked/blocked, need to unlock
   change_pin = want to change PIN to a new one
   - Query: "How many times can I enter a wrong PIN before it is blocked?" → 49
   - Query: "I entered in the wrong PIN.  Please help me unlock my account." → 49
   - Query: "Where do I need to go to change my PIN?" → 21
   - Query: "I really need to know how to change my pin." → 21

4. pending_cash_withdrawal (ID 46) vs declined_cash_withdrawal (ID 26) vs cash_withdrawal_not_recognised (ID 20):
   pending_cash_withdrawal = withdrawal is processing/pending
   declined_cash_withdrawal = withdrawal was rejected/declined
   cash_withdrawal_not_recognised = withdrawal not showing in account
   - Query: "I tried to get cash at some ATM in the city centre earlier but the machine declined my card. I've seen it still shows up as pending in my account. Please cancel it immediately as I definitely have not received that money!" → 46
   - Query: "I attempted to get cash in the ATM but it was not authorized" → 26
   - Query: "My app says I withdraw money from my account from an ATM." → 20

5. verify_my_identity (ID 69) vs why_verify_identity (ID 74) vs unable_to_verify_identity (ID 68):
   verify_my_identity = want to verify/complete verification
   why_verify_identity = asking why verification is needed
   unable_to_verify_identity = having trouble completing verification
   - Query: "If I'm getting my identity verified, what all do I need?" → 69
   - Query: "I do not feel comfortable verifying my identity." → 74
   - Query: "My identity wasn't verified" → 68

6. card_payment_wrong_exchange_rate (ID 17) vs wrong_exchange_rate_for_cash_withdrawal (ID 76) vs exchange_rate (ID 32):
   card_payment_wrong_exchange_rate = wrong rate used for CARD payment
   wrong_exchange_rate_for_cash_withdrawal = wrong rate used for CASH withdrawal
   exchange_rate = asking about current/general exchange rates
   - Query: "I seem to have been charged to much for my holiday purchases, the exchange rate is wrong." → 17
   - Query: "The exchange rate for foreign ATM currency is wrong." → 76
   - Query: "How did you guys get your exchange rate?" → 32

7. extra_charge_on_statement (ID 34) vs card_payment_fee_charged (ID 15):
   extra_charge_on_statement = unexpected charge on statement
   card_payment_fee_charged = fee charged for card payment
   - Query: "I was overcharged one extra pound!" → 34
   - Query: "There was an extra fee when I paid with my card, why was i charged this extra fee?" → 15

8. getting_virtual_card (ID 40):
   - Query: "Where do I have access to a virtual card?" → 40
   - Query: "How can I receive a virtual card?" → 40


Remember: Respond with ONLY the numeric ID, nothing else."""


def add_system_prompt_to_file(input_file, output_file):
    """Add system prompt to entries that don't have it."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    total_count = 0
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        print(f"Reading data from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    total_count += 1
                    
                    # Only process entries with "messages" key
                    if "messages" in entry:
                        messages = entry["messages"]
                        has_system = any(msg.get("role") == "system" for msg in messages)
                        
                        if not has_system:
                            # Add system prompt at the beginning
                            entry["messages"] = [
                                {"role": "system", "content": SYSTEM_PROMPT}
                            ] + messages
                            added_count += 1
                        
                        # Write the entry (whether modified or not)
                        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    else:
                        # Entry doesn't have messages, skip it
                        skipped_count += 1
                        print(f"Line {line_num}: Skipped entry (no 'messages' key)")
                        
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Line {line_num}: JSON decode error: {e}")
                except Exception as e:
                    error_count += 1
                    print(f"Line {line_num}: Error: {e}")
    
    print(f"\nProcessing complete!")
    print(f"  Total entries processed: {total_count}")
    print(f"  System prompts added: {added_count}")
    print(f"  Entries skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/banking77_train.jsonl"
    output_file = "data/banking77_train_with_system.jsonl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    add_system_prompt_to_file(input_file, output_file)


"""
Human Review Interface for Scope Labels

Interactive CLI tool for reviewing and correcting scope labels.

Usage:
    python evaluation/review_scope_labels.py \
        --input data/scope_benchmark_review.jsonl \
        --output data/scope_benchmark_review_updated.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


class ScopeReviewer:
    """Interactive reviewer for scope labels."""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.items = []
        self.current_index = 0
        self.stats = {"reviewed": 0, "changed": 0, "skipped": 0}

    def load_items(self):
        """Load review items from JSONL."""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))

        print(f"Loaded {len(self.items)} items")

    def save_items(self):
        """Save reviewed items to JSONL."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in self.items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n💾 Saved {len(self.items)} items to {self.output_path}")

    def display_item(self, item: Dict):
        """Display item for review."""
        print("\n" + "=" * 70)
        print(f"Item {self.current_index + 1}/{len(self.items)}")
        print("=" * 70)

        print(f"\n📝 Query ID: {item['query_id']}")
        print(f"\n🔍 Query:")
        print(f"  {item['query']}")

        print(f"\n🏷️  Suggested Label: {item['suggested_label']}")

        if item.get('human_label'):
            print(f"✅ Human Label: {item['human_label']}")

        print(f"\n🔧 Components:")
        for comp_name, comp_value in item['components'].items():
            if comp_value:
                print(f"  [{comp_name[0].upper()}] {comp_name:10s}: {comp_value}")
            else:
                print(f"  [ ] {comp_name:10s}: (none)")

        print(f"\n📄 Source Context:")
        context = item['source_context']
        if len(context) > 200:
            print(f"  {context[:200]}...")
        else:
            print(f"  {context}")

        if item.get('notes'):
            print(f"\n📝 Notes: {item['notes']}")

    def get_user_input(self) -> str:
        """Get user's label decision."""
        print(f"\n" + "-" * 70)
        print("Options:")
        print("  [1] TOO_BROAD    - Only domain, no specific task/problem")
        print("  [2] SEARCHABLE   - Has domain + task, or other valid combination")
        print("  [3] TOO_NARROW   - Too many constraints, almost no papers")
        print("  [a] ACCEPT       - Accept suggested label")
        print("  [s] SKIP         - Skip for now")
        print("  [n] ADD NOTE     - Add a note")
        print("  [q] QUIT         - Save and quit")
        print("-" * 70)

        choice = input("Your choice: ").strip().lower()
        return choice

    def review_item(self, item: Dict) -> bool:
        """
        Review a single item.

        Returns:
            True to continue, False to quit
        """
        self.display_item(item)

        while True:
            choice = self.get_user_input()

            if choice == '1':
                item['human_label'] = 'TOO_BROAD'
                self.stats['reviewed'] += 1
                if item['human_label'] != item['suggested_label']:
                    self.stats['changed'] += 1
                print(f"✅ Marked as TOO_BROAD")
                return True

            elif choice == '2':
                item['human_label'] = 'SEARCHABLE'
                self.stats['reviewed'] += 1
                if item['human_label'] != item['suggested_label']:
                    self.stats['changed'] += 1
                print(f"✅ Marked as SEARCHABLE")
                return True

            elif choice == '3':
                item['human_label'] = 'TOO_NARROW'
                self.stats['reviewed'] += 1
                if item['human_label'] != item['suggested_label']:
                    self.stats['changed'] += 1
                print(f"✅ Marked as TOO_NARROW")
                return True

            elif choice == 'a':
                item['human_label'] = item['suggested_label']
                self.stats['reviewed'] += 1
                print(f"✅ Accepted suggested label: {item['suggested_label']}")
                return True

            elif choice == 's':
                self.stats['skipped'] += 1
                print(f"⏭️  Skipped")
                return True

            elif choice == 'n':
                note = input("Enter note: ").strip()
                item['notes'] = note
                print(f"📝 Note added")
                # Don't advance, stay on this item

            elif choice == 'q':
                print(f"\n💾 Saving and quitting...")
                return False

            else:
                print(f"❌ Invalid choice. Please try again.")

    def run(self):
        """Run interactive review session."""
        print("=" * 70)
        print("Scope Label Review")
        print("=" * 70)
        print(f"\nInput:  {self.input_path}")
        print(f"Output: {self.output_path}")

        self.load_items()

        # Start from first unreviewed item
        for i, item in enumerate(self.items):
            if not item.get('human_label'):
                self.current_index = i
                break

        print(f"\nStarting from item {self.current_index + 1}/{len(self.items)}")
        print("\nPress Ctrl+C at any time to save and quit")

        try:
            while self.current_index < len(self.items):
                item = self.items[self.current_index]

                # Skip already reviewed items
                if item.get('human_label') and item['human_label'] != '':
                    print(f"\n[{self.current_index + 1}/{len(self.items)}] Already reviewed, skipping...")
                    self.current_index += 1
                    continue

                should_continue = self.review_item(item)

                if not should_continue:
                    break

                self.current_index += 1

                # Auto-save every 10 items
                if self.current_index % 10 == 0:
                    self.save_items()
                    print(f"✅ Auto-saved at item {self.current_index}")

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")

        # Final save
        self.save_items()

        # Print stats
        print("\n" + "=" * 70)
        print("Review Statistics")
        print("=" * 70)
        print(f"  Reviewed:       {self.stats['reviewed']}")
        print(f"  Changed:        {self.stats['changed']}")
        print(f"  Skipped:        {self.stats['skipped']}")
        print(f"  Total progress: {self.current_index}/{len(self.items)}")

        # Count completed
        completed = sum(1 for item in self.items if item.get('human_label'))
        print(f"  Completed:      {completed}/{len(self.items)} ({completed/len(self.items)*100:.1f}%)")


def generate_review_summary(input_path: str):
    """Generate summary statistics of review progress."""
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))

    print("\n" + "=" * 70)
    print("Review Summary")
    print("=" * 70)

    total = len(items)
    reviewed = sum(1 for item in items if item.get('human_label'))
    changed = sum(
        1 for item in items
        if item.get('human_label') and item['human_label'] != item['suggested_label']
    )

    print(f"\nProgress: {reviewed}/{total} ({reviewed/total*100:.1f}%)")
    print(f"Changed:  {changed}/{reviewed if reviewed > 0 else 1} ({changed/(reviewed if reviewed > 0 else 1)*100:.1f}%)")

    # Label distribution
    label_counts = {}
    for item in items:
        if item.get('human_label'):
            label = item['human_label']
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nHuman Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15s}: {count:3d} ({count/reviewed*100:.1f}%)")

    # Agreement with suggested labels
    if reviewed > 0:
        agreement = reviewed - changed
        print(f"\nAgreement with Suggested Labels:")
        print(f"  Agree:    {agreement:3d} ({agreement/reviewed*100:.1f}%)")
        print(f"  Disagree: {changed:3d} ({changed/reviewed*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive review tool for scope labels"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Review command
    review_parser = subparsers.add_parser("review", help="Start review session")
    review_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with items to review"
    )
    review_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file (default: input file with _updated suffix)"
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show review summary")
    summary_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSONL file to summarize"
    )

    args = parser.parse_args()

    if args.command == "review":
        # Default output path
        if not args.output:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_updated.jsonl")

        reviewer = ScopeReviewer(args.input, args.output)
        reviewer.run()

    elif args.command == "summary":
        generate_review_summary(args.input)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

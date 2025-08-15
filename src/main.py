import argparse
from src.scraping.scraper import scrape_and_save_state_llc_data

DEMO_STATES = ["California", "Texas"]

if __name__ == "__main__":
    p = argparse.ArgumentParser("Scrape Bizee LLC pages")
    p.add_argument("--states", nargs="+", help="e.g. --states California Texas")
    p.add_argument("--all-demo", action="store_true", help="Scrape a demo set")
    args = p.parse_args()

    states = DEMO_STATES if args.all_demo else (args.states or [])
    if not states:
        print("No states given. Use --states or --all-demo.")
        raise SystemExit(1)

    for s in states:
        scrape_and_save_state_llc_data(s)

#!/usr/bin/env python
"""Console interface for the music recommender.

    python cli.py                      interactive menu
    python cli.py search "daft punk"
    python cli.py similar "blinding lights" -k 15
    python cli.py mix "get lucky" "redbone" "september"
    python cli.py mood --energy 0.8 --valence 0.8 --tempo 124 --genre disco
    python cli.py playlist 42
"""
import argparse
import os
import shutil
import signal
import sys
import unicodedata

if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet exit when piped into head

BOLD, DIM, GREEN, CYAN, RESET = ("\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[0m") \
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else ("",) * 5

METHODS = ["hybrid", "content", "knn", "als", "popularity"]


def load():
    from src import config
    if not config.MODELS.exists():
        sys.exit("No trained models found. Run `python pipeline.py` (about 2 minutes) or the notebooks first.")
    print(f"{DIM}Loading catalog and models...{RESET}", end="\r", flush=True)
    from src.recommender import MusicRecommender
    r = MusicRecommender()
    print(" " * 40, end="\r")
    return r


def _w(ch):
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def clip(s, n):
    """Cut and pad to n terminal columns (CJK characters take two)."""
    total = sum(_w(c) for c in s)
    if total > n:
        out, width = "", 0
        for c in s:
            if width + _w(c) > n - 1:
                break
            out += c; width += _w(c)
        s, total = out + "…", width + 1
    return s + " " * (n - total)


def print_tracks(rec, items, seeds=None, show_why=True):
    width = shutil.get_terminal_size((110, 20)).columns
    title_w = max(22, min(40, (width - 50) // 2))
    artist_w = max(16, min(28, width - title_w - 52))
    print(f"{DIM}{'#':>3}  {'Title':<{title_w}} {'Artist':<{artist_w}} {'Genre':<14} {'BPM':>4}  {'Key':<9}{RESET}")
    for n, item in enumerate(items, 1):
        i = item[0] if isinstance(item, tuple) else item
        d = rec.describe(i)
        print(f"{n:>3}  {BOLD}{clip(d['title'], title_w)}{RESET} {clip(d['artist'], artist_w)} "
              f"{CYAN}{clip(d['genre'].split(', ')[0], 14)}{RESET} {d['bpm']:>4}  {d['key']:<9}")
        if show_why and seeds:
            print(f"     {GREEN}↳ {' · '.join(rec.explain(seeds, i))}{RESET}")
    print()


def pick(rec, query, interactive=True):
    hits = rec.search(query, k=8)
    if not hits:
        print(f"No song matches “{query}”. Try the artist's name, or fewer words.")
        return None
    if len(hits) == 1 or not interactive:
        return hits[0]
    print_tracks(rec, hits, show_why=False)
    choice = input(f"Which one? [1-{len(hits)}, Enter = 1] ").strip()
    if not choice:
        return hits[0]
    return hits[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(hits) else None


def show_recs(rec, seeds, k=10, method="hybrid", diversity=0.2, new_artists=False):
    names = ", ".join(rec._short(rec.describe(s)["title"], 40) for s in seeds[:3])
    if len(seeds) > 3:
        names += f" and {len(seeds) - 3} more"
    print(f"\n{BOLD}Because you like {names}{RESET}  {DIM}({method}){RESET}\n")
    recs = rec.recommend(seeds, k=k, method=method, diversity=diversity, exclude_seed_artists=new_artists)
    print_tracks(rec, recs, seeds)
    return recs


def interactive(rec):
    seeds, settings = [], {"k": 10, "method": "hybrid", "diversity": 0.2, "new_artists": False}
    menu = f"""{BOLD}What next?{RESET}
  {BOLD}1{RESET} Find songs like one song      {BOLD}4{RESET} See a listener's playlist + picks
  {BOLD}2{RESET} Mix from several songs        {BOLD}5{RESET} Settings (method, length, variety)
  {BOLD}3{RESET} Search by mood                {BOLD}q{RESET} Quit"""
    print(f"\n{BOLD}Music Recommender{RESET}  {DIM}{rec.n:,} tracks · {len(rec.genre_sim)} genres · "
          f"{len(rec.playlists):,} simulated listeners{RESET}")
    while True:
        print(menu)
        c = input("> ").strip().lower()
        try:
            if c == "1":
                i = pick(rec, input("Song or artist: "))
                if i is not None:
                    show_recs(rec, [i], **settings)
            elif c == "2":
                seeds = []
                print("Add songs one at a time. Press Enter on an empty line when done.")
                while len(seeds) < 10:
                    q = input(f"Song {len(seeds) + 1}: ").strip()
                    if not q:
                        break
                    i = pick(rec, q)
                    if i is not None:
                        seeds.append(i); print(f"  {GREEN}added {rec.describe(i)['title']}{RESET}")
                if seeds:
                    show_recs(rec, seeds, **settings)
            elif c == "3":
                print("Rate each from 0 to 10 (Enter skips it).")
                t = {}
                for dim, label in [("energy", "Energy (calm → intense)"), ("valence", "Mood (sad → happy)"),
                                   ("danceability", "Danceability"), ("acousticness", "Acoustic (electronic → unplugged)")]:
                    v = input(f"  {label}: ").strip()
                    if v: t[dim] = float(v) / 10
                v = input("  Tempo in BPM: ").strip()
                if v: t["tempo"] = float(v)
                g = input("  Genre (Enter = any): ").strip().lower() or None
                if g and g not in rec.genre_sim.index:
                    close = [x for x in rec.genre_sim.index if g in x]
                    print(f"  Unknown genre. {'Did you mean ' + ', '.join(close[:5]) + '?' if close else ''}")
                    continue
                if not t:
                    print("Set at least one value."); continue
                print()
                print_tracks(rec, rec.mood(t, genre=g, k=settings["k"]))
            elif c == "4":
                v = input(f"Listener number 0-{len(rec.playlists) - 1} (Enter = random): ").strip()
                pid = int(v) if v.isdigit() else int(__import__("random").randrange(len(rec.playlists)))
                show_playlist(rec, pid, settings)
            elif c == "5":
                v = input(f"Method {METHODS} [{settings['method']}]: ").strip().lower()
                if v in METHODS: settings["method"] = v
                v = input(f"How many songs [{settings['k']}]: ").strip()
                if v.isdigit(): settings["k"] = max(1, min(50, int(v)))
                v = input(f"Variety 0-10 [{int(settings['diversity'] * 10)}]: ").strip()
                if v.isdigit(): settings["diversity"] = min(10, int(v)) / 10
                v = input(f"Only new artists? y/n [{'y' if settings['new_artists'] else 'n'}]: ").strip().lower()
                if v in ("y", "n"): settings["new_artists"] = v == "y"
                print(f"{GREEN}Saved.{RESET}\n")
            elif c in ("q", "quit", "exit"):
                return
        except (ValueError, IndexError):
            print("That input didn't work. Try again.\n")


def show_playlist(rec, pid, settings=None):
    settings = settings or {"k": 10, "method": "hybrid", "diversity": 0.2, "new_artists": False}
    p = rec.playlists.iloc[pid]
    items = rec.playlist_tracks(pid)
    print(f"\n{BOLD}{p.playlist_name}{RESET}  {DIM}{len(items)} tracks · taste: {p.taste_genres.replace(';', ', ')}{RESET}\n")
    print_tracks(rec, items[:8], show_why=False)
    if len(items) > 8:
        print(f"{DIM}     … and {len(items) - 8} more{RESET}\n")
    show_recs(rec, items, **settings)


def main():
    ap = argparse.ArgumentParser(description="Music recommender (console)")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("search", help="find songs"); s.add_argument("query")
    for name, help_ in [("similar", "songs like one song"), ("mix", "songs like several songs")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("songs", nargs="+")
        p.add_argument("-k", type=int, default=10)
        p.add_argument("--method", choices=METHODS, default="hybrid")
        p.add_argument("--variety", type=float, default=0.2, help="0 = closest matches, 1 = most varied")
        p.add_argument("--new-artists", action="store_true", help="leave out the seed artists")
    m = sub.add_parser("mood", help="songs matching a mood")
    for f in ["energy", "valence", "danceability", "acousticness"]:
        m.add_argument(f"--{f}", type=float, help="0 to 1")
    m.add_argument("--tempo", type=float, help="BPM"); m.add_argument("--genre"); m.add_argument("-k", type=int, default=10)
    pl = sub.add_parser("playlist", help="a simulated listener's playlist and recommendations")
    pl.add_argument("pid", type=int); pl.add_argument("-k", type=int, default=10)
    a = ap.parse_args()

    rec = load()
    if a.cmd is None:
        try:
            interactive(rec)
        except (KeyboardInterrupt, EOFError):
            print()
    elif a.cmd == "search":
        print_tracks(rec, rec.search(a.query), show_why=False)
    elif a.cmd in ("similar", "mix"):
        queries = a.songs[:1] if a.cmd == "similar" else a.songs
        seeds = [i for i in (pick(rec, q, interactive=False) for q in queries) if i is not None]
        if seeds:
            show_recs(rec, seeds, k=a.k, method=a.method, diversity=a.variety, new_artists=a.new_artists)
    elif a.cmd == "mood":
        t = {k: v for k, v in vars(a).items() if k in ("energy", "valence", "danceability", "acousticness", "tempo") and v is not None}
        if not t:
            sys.exit("Give at least one of --energy --valence --danceability --acousticness --tempo")
        print_tracks(rec, rec.mood(t, genre=a.genre, k=a.k))
    elif a.cmd == "playlist":
        show_playlist(rec, a.pid, {"k": a.k, "method": "hybrid", "diversity": 0.2, "new_artists": False})


if __name__ == "__main__":
    main()

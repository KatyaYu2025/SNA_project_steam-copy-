#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Friend Prediction - Open World Games Data Collector (OPTIMIZED)
优化重点:
  1. 移除种子预筛选，避免高频请求触发限流
  2. 智能指数退避 + 限流计数熔断
  3. 简易内存缓存，减少重复请求
  4. 更详细的进度与诊断日志
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Friend Prediction - Open World Games Data Collector (FINAL FIX)
修复清单:
  1. 删除错误的 `import seen` 
  2. 修复 requests.get 参数问题
  3. 初始化 seen/queued_ids 集合
  4. 统一 timeout 配置传递
  5. 优化日志级别，让等待过程可见
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Friend Prediction - Open World Games Data Collector (FINAL FIXED)
修复清单:
  1. 删除错误的 import seen
  2. 修复 safe_request 参数传递问题
  3. 正确初始化 seen/queued_ids 集合
  4. 修改筛选条件: 单个目标游戏时长 < 2000 小时（非总时长）
  5. 限流日志改为 info 级别，可见等待过程
"""

import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import os
import sys
from collections import deque
from typing import List, Set, Optional, Dict, Any

# ================= 配置区 =================
STEAM_API_KEY = "78C59CE7EC039BF4B9293F61AE8C9A6A"  # 🔴 替换为你的 32 位 Key

# 🎯 采集参数
TARGET_PLAYERS = 500  # 目标合格玩家数
REQUEST_DELAY = 2.0  # 基础请求间隔 (秒)
MAX_RETRIES = 2  # 单请求最大重试
MIN_TARGET_GAMES = 2  # 最少目标游戏数

# ⏱️ 筛选阈值 - ✅ 修改: 单个游戏时长限制
MIN_PLAYTIME_HOURS = 1  # 单个目标游戏最小时长
MAX_PLAYTIME_HOURS_SINGLE = 2000  # ✅ 单个目标游戏最大时长（原为总时长）
MAX_FRIENDS_COUNT = 500  # 最大好友数量

# 🚨 限流熔断配置
RATE_LIMIT_THRESHOLD = 10  # 连续 429 次数阈值
RATE_LIMIT_COOLDOWN = 60  # 触发熔断后等待秒数

# 🎮 目标游戏 (15 款)
TARGET_GAMES = {
    1245620: "Elden Ring",
    1091500: "Cyberpunk 2077",
    990080: "Hogwarts Legacy",
    1174180: "Red Dead Redemption 2",
    271590: "Grand Theft Auto V",
    489830: "The Elder Scrolls V: Skyrim Special Edition",
    292030: "The Witcher 3: Wild Hunt",
    105600: "Terraria",
    413150: "Stardew Valley",
    264710: "Subnautica",
    332200: "State of Decay 2",
    377160: "Fallout 4",
    275850: "No Man's Sky",
    252490: "Rust",
    534380: "Dying Light 2 Stay Human",
}

# 🌱 种子组
SEED_GROUPS = ["OpenWorldGaming", "RPGGaming", "SurvivalGames", "SteamUniverse", "PCMasterRace"]

OUTPUT_DIR = "data_openworld"
LOG_FILE = f"{OUTPUT_DIR}/collector.log"

# ==========================================

# 全局缓存与限流计数
_player_cache: Dict[str, Dict[str, Any]] = {}
_rate_limit_count = 0


def setup_logging():
    """配置日志输出"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if root.handlers:
        root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))

    file_h = logging.FileHandler(LOG_FILE, 'w', encoding='utf-8')
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%H:%M:%S'))

    root.addHandler(console)
    root.addHandler(file_h)
    return root


logger = None


def _handle_rate_limit():
    """限流熔断处理 - 日志改为 info 级别"""
    global _rate_limit_count
    _rate_limit_count += 1
    if _rate_limit_count >= RATE_LIMIT_THRESHOLD:
        logger.warning(f"🚨 触发限流熔断! 等待 {RATE_LIMIT_COOLDOWN} 秒...")
        time.sleep(RATE_LIMIT_COOLDOWN)
        _rate_limit_count = 0
    else:
        wait = REQUEST_DELAY * (2 ** min(_rate_limit_count, 4))
        # ✅ 改为 info，让控制台可见
        logger.info(f"⏳ 限流退避: 等待 {wait:.1f} 秒 (连续 {_rate_limit_count} 次)")
        time.sleep(wait)


def safe_request(url: str, params: Optional[dict] = None, timeout: tuple = (5, 10),
                 steamid: Optional[str] = None) -> Optional[requests.Response]:
    """
    增强版请求函数
    ✅ 修复: 移除传递给 requests 的 steamid 参数，仅用于日志
    """
    global _rate_limit_count

    for attempt in range(MAX_RETRIES):
        try:
            # ✅ 修复: timeout 使用元组 (connect, read)，不传递 steamid 给 requests
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            _rate_limit_count = 0
            return resp

        except requests.exceptions.HTTPError as e:
            resp_obj = e.response if hasattr(e, 'response') else None
            if resp_obj:
                if resp_obj.status_code == 429:
                    _handle_rate_limit()
                    continue
                elif resp_obj.status_code in [401, 403]:
                    sid_log = f"{steamid[:10]}..." if steamid else "???"
                    logger.debug(f"🔒 隐私限制 {sid_log}: {resp_obj.status_code}")
                    return "PRIVATE"
                else:
                    logger.debug(f"⚠️ HTTP {resp_obj.status_code}: {url[:60]}")
            else:
                logger.debug(f"⚠️ HTTPError 无 response")

        except requests.exceptions.Timeout:
            logger.debug(f"⏱️ 超时 ({attempt + 1}/{MAX_RETRIES}): {url[:50]}")
        except requests.exceptions.RequestException as ex:
            logger.debug(f"🔍 请求异常: {type(ex).__name__}: {str(ex)[:40]}")

        time.sleep(REQUEST_DELAY * (attempt + 1))

    return None


def fetch_group_members(group_id: str) -> List[str]:
    """获取组成员列表"""
    url = f"https://steamcommunity.com/groups/{group_id}/memberslistxml/?xml=1"
    resp = safe_request(url, timeout=(5, 20), steamid=None)
    if not resp or resp == "PRIVATE":
        logger.warning(f"⚠️ 组 '{group_id}' 获取失败")
        return []
    try:
        root = ET.fromstring(resp.content)
        members = [m.text for m in root.findall(".//steamID64") if m.text and m.text.isdigit()]
        logger.info(f"✅ 组 '{group_id}' | 获取 {len(members)} 个种子 ID")
        return members
    except Exception as e:
        logger.error(f"❌ 解析组 '{group_id}' 失败: {e}")
        return []


def _get_from_cache(steamid: str, key: str) -> Optional[Any]:
    """缓存读取 (1 小时有效期)"""
    cached = _player_cache.get(steamid)
    if cached and key in cached:
        if time.time() - cached.get("timestamp", 0) < 3600:
            return cached[key]
    return None


def _set_cache(steamid: str, key: str, value: Any):
    """缓存写入"""
    _player_cache.setdefault(steamid, {"timestamp": time.time()})
    _player_cache[steamid][key] = value


def get_owned_games(steamid: str) -> Optional[List[dict]]:
    """获取游戏库"""
    cached = _get_from_cache(steamid, "games")
    if cached is not None:
        return cached if cached != "PRIVATE" else "PRIVATE"

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {"key": STEAM_API_KEY, "steamid": steamid, "include_appinfo": 1, "include_played_free_games": 1}
    resp = safe_request(url, params=params, timeout=(5, 12), steamid=steamid)

    if resp == "PRIVATE":
        _set_cache(steamid, "games", "PRIVATE")
        return "PRIVATE"
    if resp is None:
        return None

    try:
        data = resp.json()
        games = []
        for g in data.get("response", {}).get("games", []):
            if "appid" in g:
                games.append({
                    "appid": int(g["appid"]),
                    "playtime_forever": int(g.get("playtime_forever", 0)),
                    "name": g.get("name", "Unknown")
                })
        _set_cache(steamid, "games", games)
        return games
    except Exception:
        return []


def get_friends(steamid: str) -> Optional[List[str]]:
    """获取好友列表"""
    cached = _get_from_cache(steamid, "friends")
    if cached is not None:
        return cached if cached != "PRIVATE" else "PRIVATE"

    url = "https://api.steampowered.com/ISteamUser/GetFriendList/v0001/"
    params = {"key": STEAM_API_KEY, "steamid": steamid, "relationship": "friend"}
    resp = safe_request(url, params=params, timeout=(5, 12), steamid=steamid)

    if resp == "PRIVATE":
        _set_cache(steamid, "friends", "PRIVATE")
        return "PRIVATE"
    if resp is None:
        return None

    try:
        data = resp.json()
        friends_list = data.get("friendslist", {}).get("friends", []) or data.get("friends", [])
        friends = [f["steamid"] for f in friends_list if isinstance(f, dict) and "steamid" in f]
        _set_cache(steamid, "friends", friends)
        return friends
    except Exception:
        return []


def check_qualification(games: List[dict], friends_count: int) -> Dict[str, Any]:
    """
    检查玩家是否符合筛选条件
    ✅ 修改: 单个目标游戏时长 < 2000 小时（非总时长）
    """
    owned_target = {g["appid"]: g for g in games if g["appid"] in TARGET_GAMES}
    count = len(owned_target)

    if count < MIN_TARGET_GAMES:
        return {"ok": False, "reason": f"目标游戏{count}<{MIN_TARGET_GAMES}", "target": owned_target}

    # ✅ 修改: 检查单个游戏时长，而非总时长
    for appid, game in owned_target.items():
        playtime_h = game["playtime_forever"] / 60
        if playtime_h < MIN_PLAYTIME_HOURS:
            return {"ok": False, "reason": f"{TARGET_GAMES[appid]}时长{playtime_h:.1f}h<{MIN_PLAYTIME_HOURS}h",
                    "target": owned_target, "playtime_h": playtime_h}
        if playtime_h > MAX_PLAYTIME_HOURS_SINGLE:
            return {"ok": False, "reason": f"{TARGET_GAMES[appid]}时长{playtime_h:.1f}h>{MAX_PLAYTIME_HOURS_SINGLE}h",
                    "target": owned_target, "playtime_h": playtime_h}

    if friends_count > MAX_FRIENDS_COUNT:
        return {"ok": False, "reason": f"好友{friends_count}>{MAX_FRIENDS_COUNT}",
                "target": owned_target, "playtime_h": sum(g["playtime_forever"] for g in owned_target.values()) / 60}

    # 计算平均时长用于展示
    avg_playtime = sum(g["playtime_forever"] for g in owned_target.values()) / 60 / count if count > 0 else 0

    return {
        "ok": True,
        "reason": "OK",
        "target": {
            "count": count,
            "games": owned_target,
            "playtime_h": avg_playtime,  # 展示用平均值
            "names": [TARGET_GAMES[appid] for appid in owned_target]
        }
    }


def validate_api_key() -> bool:
    """验证 API Key"""
    if STEAM_API_KEY in ["YOUR_STEAM_API_KEY_HERE", ""] or len(STEAM_API_KEY) != 32:
        logger.error("🔴 请替换脚本顶部的 STEAM_API_KEY (需 32 位有效 Key)")
        return False
    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    resp = safe_request(url, params={"key": STEAM_API_KEY, "steamids": "76561197960287930"}, timeout=(5, 10))
    if not resp or resp == "PRIVATE":
        logger.error("🔴 API 连接测试失败")
        return False
    try:
        data = resp.json()
        if "players" not in data.get("response", {}):
            logger.error(f"🔴 API Key 无效: {data}")
            return False
        logger.info("✅ API Key 验证通过")
        return True
    except Exception as e:
        logger.error(f"🔴 解析失败: {e}")
        return False


def main():
    global logger, _rate_limit_count
    logger = setup_logging()

    print(">>> 🚀 开放世界玩家采集器 (最终修复版) 启动", flush=True)
    logger.info("🎮 目标游戏: %d 款 | 🎯 目标玩家: %d", len(TARGET_GAMES), TARGET_PLAYERS)
    logger.info("📋 筛选: ≥%d 款目标游戏 | 单游戏时长 %.1f~%.1f 小时 | 好友≤%d",
                MIN_TARGET_GAMES, MIN_PLAYTIME_HOURS, MAX_PLAYTIME_HOURS_SINGLE, MAX_FRIENDS_COUNT)

    if not validate_api_key():
        print(">>> ❌ API Key 验证失败", flush=True)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 📊 统计计数器
    stats = {
        "total_checked": 0, "passed": 0,
        "rejected_game": 0, "rejected_time": 0, "rejected_friends": 0,
        "rejected_privacy": 0, "rejected_api": 0,
        "rate_limit_hits": 0,
        "target_dist": {appid: 0 for appid in TARGET_GAMES}
    }

    # ✅ 修复: 在 main() 内初始化所有集合
    seen = set()  # 种子收集阶段去重
    queued_ids = set()  # 主循环入队查重 (O(1) 查找)

    # 🌱 步骤 1: 收集种子玩家 ID
    logger.info("🌱 收集种子玩家 ID (仅组成员列表)...")
    seed_queue = deque()

    for group in SEED_GROUPS:
        members = fetch_group_members(group)
        for sid in members:
            if sid not in seen and sid not in queued_ids:
                seen.add(sid)
                seed_queue.append(sid)
                queued_ids.add(sid)
        time.sleep(1.0)

    logger.info(f"📦 种子池: {len(seed_queue)} 个唯一玩家 ID")
    if len(seed_queue) == 0:
        logger.error("❌ 无种子玩家，检查组名或网络")
        return

    # 🗄️ 数据容器
    collected = set()  # 合格玩家
    processed = set()  # 已检查玩家
    edges = []
    player_games = []
    profiles = []

    # 🔄 步骤 2: 主循环筛选 + 采集
    logger.info("🔄 开始主循环筛选 (带缓存 + 限流保护)...")
    start = time.time()

    while len(collected) < TARGET_PLAYERS and seed_queue:
        sid = seed_queue.popleft()

        if sid in collected or sid in processed:
            continue
        processed.add(sid)
        stats["total_checked"] += 1

        # 获取游戏库
        games = get_owned_games(sid)
        if games == "PRIVATE":
            stats["rejected_privacy"] += 1
            continue
        if games is None:
            stats["rejected_api"] += 1
            continue
        if not games:
            continue

        # 获取好友列表
        friends = get_friends(sid)
        if friends == "PRIVATE":
            stats["rejected_privacy"] += 1
            continue
        if friends is None:
            stats["rejected_api"] += 1
            continue

        # 筛选检查
        result = check_qualification(games, len(friends))
        if not result["ok"]:
            r = result["reason"]
            if "目标游戏" in r:
                stats["rejected_game"] += 1
            elif "时长" in r:
                stats["rejected_time"] += 1
            elif "好友" in r:
                stats["rejected_friends"] += 1
            logger.debug(f"⏭️ 跳过 {sid}: {r}")
            continue

        # ✅ 采集合格玩家
        progress = len(collected) + 1
        elapsed = time.time() - start
        eta = (elapsed / progress) * (TARGET_PLAYERS - progress) if progress < TARGET_PLAYERS else 0
        t = result["target"]

        print(f">>> 🎮 [{progress}/{TARGET_PLAYERS}] {sid} | "
              f"游戏: {t['names']} | 平均时长: {t['playtime_h']:.1f}h | 好友: {len(friends)} | ETA: {eta:.0f}s",
              flush=True)

        stats["passed"] += 1
        for appid in t["games"]:
            stats["target_dist"][appid] += 1

        # 记录玩家信息
        profiles.append({
            "steamid": sid,
            "target_games_count": t["count"],
            "avg_target_playtime_h": round(t["playtime_h"], 2),
            "friends_count": len(friends),
            "total_games": len(games)
        })

        # 好友入队（雪球采样）
        for fid in friends:
            if fid not in collected and fid not in processed and fid not in queued_ids:
                seed_queue.append(fid)
                queued_ids.add(fid)  # ✅ 关键: 同步加入查重集合
            p1, p2 = sorted([sid, fid])
            edges.append({"player1": p1, "player2": p2})

        # 记录玩家-游戏关系
        for g in games:
            player_games.append({
                "steamid": sid,
                "appid": g["appid"],
                "playtime_forever": g["playtime_forever"],
                "game_name": g["name"],
                "is_target": g["appid"] in TARGET_GAMES
            })

        collected.add(sid)
        time.sleep(REQUEST_DELAY)

    # 💾 保存数据
    logger.info("💾 保存数据到 CSV...")

    if profiles:
        pd.DataFrame(profiles).to_csv(f"{OUTPUT_DIR}/players.csv", index=False, encoding='utf-8-sig')
        logger.info(f"✅ 玩家列表: {len(profiles)} 条")

    if edges:
        pd.DataFrame(edges).drop_duplicates().to_csv(f"{OUTPUT_DIR}/friend_edges.csv", index=False,
                                                     encoding='utf-8-sig')
        logger.info(f"✅ 好友边: {len(edges)} 条 (去重后)")

    if player_games:
        pd.DataFrame(player_games).to_csv(f"{OUTPUT_DIR}/player_games.csv", index=False, encoding='utf-8-sig')
        logger.info(f"✅ 玩家-游戏边: {len(player_games)} 条")

    # 保存统计
    stats["elapsed"] = round(time.time() - start, 1)
    stats["pass_rate"] = round(stats["passed"] / stats["total_checked"] * 100, 2) if stats["total_checked"] else 0
    pd.DataFrame([stats]).to_csv(f"{OUTPUT_DIR}/collection_stats.csv", index=False, encoding='utf-8-sig')

    # 🎉 完成报告
    logger.info("✨ 采集完成!")
    logger.info(f"   📦 合格玩家: {len(collected)}/{TARGET_PLAYERS}")
    logger.info(f"   🔗 好友边: {len(edges)} | 🎮 玩家-游戏边: {len(player_games)}")
    logger.info(f"   📊 通过率: {stats['pass_rate']}% | ⏱️ 耗时: {stats['elapsed']}s")
    logger.info(
        f"   📉 拒绝统计: 游戏={stats['rejected_game']} 时长={stats['rejected_time']} 好友={stats['rejected_friends']} 隐私={stats['rejected_privacy']} API={stats['rejected_api']}")

    if stats["rate_limit_hits"] > 0:
        logger.warning(f"   ⚠️ 触发限流 {stats['rate_limit_hits']} 次，建议增加 REQUEST_DELAY")

    # 目标游戏分布
    logger.info("🎮 目标游戏分布:")
    for appid, name in TARGET_GAMES.items():
        cnt = stats["target_dist"][appid]
        if cnt > 0:
            logger.info(f"   • {name}: {cnt} 玩家")

    print(f">>> ✅ 输出目录: {os.path.abspath(OUTPUT_DIR)}/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n>>> ⚠️ 用户中断，脚本退出", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n>>> 💥 未捕获异常: {type(e).__name__}: {e}", flush=True)
        if logger:
            logger.exception("💥 全局异常")
        sys.exit(1)
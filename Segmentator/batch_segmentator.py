#!/usr/bin/env python3
"""
批量处理ScanNet场景的segmentation生成脚本

用法:
    python batch_segmentator.py \
        --input_dir /home/kylin/datasets/scannet/scannetv2/scans \
        --output_dir /home/kylin/lyx/project_study/mask3d_data/raw/scannet_test_segments \
        --segmentator_path /path/to/ScanNet/Segmentator/segmentator \
        --kthresh 0.01 \
        --seg_min_verts 20 \
        --num_workers 4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_scene(scene_dir, output_dir, segmentator_bin, kthresh, seg_min_verts):
    """
    处理单个场景的segmentation生成
    
    Args:
        scene_dir: 场景文件夹路径 (Path对象)
        output_dir: 输出目录路径 (Path对象)
        segmentator_bin: segmentator二进制文件路径 (Path对象)
        kthresh: 分割阈值参数
        seg_min_verts: 最小顶点数
    
    Returns:
        (success: bool, scene_name: str, error_msg: str)
    """
    scene_name = scene_dir.name
    ply_file = scene_dir / f"{scene_name}_vh_clean_2.ply"
    
    # 检查PLY文件是否存在
    if not ply_file.exists():
        return False, scene_name, f"PLY file not found: {ply_file}"
    
    try:
        # 调用segmentator二进制文件
        # segmentator会在PLY文件同目录下生成 .segs.json 文件
        cmd = [
            str(segmentator_bin),
            str(ply_file),
            str(kthresh),
            str(seg_min_verts),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(ply_file.parent)
        )
        
        # segmentator生成的文件名格式: <base_name>.<kthresh>.segs.json
        # 例如: scene0707_00_vh_clean_2.0.010000.segs.json
        # segmentator可能生成不同的格式，所以我们需要查找匹配的文件
        pattern = f"{scene_name}_vh_clean_2.*.segs.json"
        candidates = list(ply_file.parent.glob(pattern))
        
        if not candidates:
            return False, scene_name, f"No segs.json file generated for {scene_name}"
        
        # 选择最匹配的文件（优先选择包含kthresh字符串的）
        kthresh_str = f"{kthresh:.6f}"
        matching = [c for c in candidates if kthresh_str in c.name]
        if matching:
            seg_file = matching[0]
        else:
            # 如果没有精确匹配，选择最新的
            seg_file = max(candidates, key=lambda p: p.stat().st_mtime)
        
        # 读取生成的segmentation文件
        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
        
        # 验证文件格式
        if "segIndices" not in seg_data:
            return False, scene_name, f"Invalid segs.json format: missing segIndices"
        
        # 创建输出文件名（Mask3D期望的格式）
        output_filename = f"{scene_name}_vh_clean_2.0.010000.segs.json"
        output_file = output_dir / output_filename
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制/移动文件到输出目录
        import shutil
        shutil.copy2(seg_file, output_file)
        
        # 可选：删除原始目录中的临时文件（如果需要）
        # seg_file.unlink()
        
        return True, scene_name, None
        
    except subprocess.CalledProcessError as e:
        return False, scene_name, f"Segmentator error: {e.stderr}"
    except Exception as e:
        return False, scene_name, f"Unexpected error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="批量处理ScanNet场景的segmentation生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录路径，包含场景文件夹（例如: /home/kylin/datasets/scannet/scannetv2/scans）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径（例如: /home/kylin/lyx/project_study/mask3d_data/raw/scannet_test_segments）"
    )
    parser.add_argument(
        "--segmentator_path",
        type=str,
        default=None,
        help="segmentator二进制文件路径。如果不指定，将尝试从脚本所在目录查找"
    )
    parser.add_argument(
        "--kthresh",
        type=float,
        default=0.01,
        help="分割阈值参数（默认: 0.01）"
    )
    parser.add_argument(
        "--seg_min_verts",
        type=int,
        default=20,
        help="每个segment的最小顶点数（默认: 20）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行处理的worker数量（默认: 4）"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已存在的输出文件"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 确定segmentator二进制文件路径
    if args.segmentator_path:
        segmentator_bin = Path(args.segmentator_path)
    else:
        # 尝试从脚本所在目录查找
        script_dir = Path(__file__).resolve().parent
        segmentator_bin = script_dir / "segmentator"
    
    if not segmentator_bin.exists():
        print(f"错误: segmentator二进制文件不存在: {segmentator_bin}")
        print(f"请先运行 'make' 编译segmentator，或使用 --segmentator_path 指定路径")
        sys.exit(1)
    
    if not segmentator_bin.is_file():
        print(f"错误: segmentator路径不是文件: {segmentator_bin}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有场景文件夹
    # 场景文件夹通常以 "scene" 开头
    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("scene")]
    
    if not scene_dirs:
        print(f"警告: 在 {input_dir} 中未找到场景文件夹")
        sys.exit(1)
    
    print(f"找到 {len(scene_dirs)} 个场景文件夹")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Segmentator: {segmentator_bin}")
    print(f"参数: kthresh={args.kthresh}, seg_min_verts={args.seg_min_verts}")
    print(f"并行workers: {args.num_workers}")
    print("-" * 60)
    
    # 过滤已存在的文件（如果设置了skip_existing）
    if args.skip_existing:
        remaining_scenes = []
        for scene_dir in scene_dirs:
            output_filename = f"{scene_dir.name}_vh_clean_2.0.010000.segs.json"
            output_file = output_dir / output_filename
            if not output_file.exists():
                remaining_scenes.append(scene_dir)
            else:
                print(f"跳过已存在的文件: {output_filename}")
        scene_dirs = remaining_scenes
        print(f"剩余需要处理的场景: {len(scene_dirs)}")
    
    # 批量处理
    success_count = 0
    fail_count = 0
    failed_scenes = []
    
    if args.num_workers > 1:
        # 并行处理
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_scene = {
                executor.submit(
                    process_scene,
                    scene_dir,
                    output_dir,
                    segmentator_bin,
                    args.kthresh,
                    args.seg_min_verts
                ): scene_dir
                for scene_dir in scene_dirs
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(scene_dirs), desc="处理场景") as pbar:
                for future in as_completed(future_to_scene):
                    scene_dir = future_to_scene[future]
                    try:
                        success, scene_name, error_msg = future.result()
                        if success:
                            success_count += 1
                            pbar.set_postfix({"成功": success_count, "失败": fail_count})
                        else:
                            fail_count += 1
                            failed_scenes.append((scene_name, error_msg))
                            pbar.set_postfix({"成功": success_count, "失败": fail_count})
                    except Exception as e:
                        fail_count += 1
                        failed_scenes.append((scene_dir.name, str(e)))
                        pbar.set_postfix({"成功": success_count, "失败": fail_count})
                    finally:
                        pbar.update(1)
    else:
        # 串行处理
        for scene_dir in tqdm(scene_dirs, desc="处理场景"):
            success, scene_name, error_msg = process_scene(
                scene_dir,
                output_dir,
                segmentator_bin,
                args.kthresh,
                args.seg_min_verts
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_scenes.append((scene_name, error_msg))
    
    # 打印总结
    print("-" * 60)
    print(f"处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    if failed_scenes:
        print("\n失败的场景:")
        for scene_name, error_msg in failed_scenes:
            print(f"  - {scene_name}: {error_msg}")


if __name__ == "__main__":
    main()

#!/bin/bash
# 网络异常检测项目启动脚本

echo "=========================================="
echo "网络异常检测项目"
echo "=========================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

echo "✓ Python3 已安装"

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "错误: pip3 未安装"
    exit 1
fi

echo "✓ pip3 已安装"

# 创建虚拟环境（可选）
if [ "$1" = "--venv" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ 虚拟环境已创建并激活"
fi

# 安装依赖
echo ""
echo "安装依赖包..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ 依赖包安装成功"
else
    echo "✗ 依赖包安装失败"
    exit 1
fi

# 启动Jupyter Notebook
echo ""
echo "启动Jupyter Notebook..."
echo "按 Ctrl+C 停止服务器"
echo ""

jupyter notebook Network_Anomaly_Detection.ipynb

echo ""
echo "项目已完成"

import cv2  # 用于图像处理的OpenCV库
import numpy as np  # 用于数值计算的NumPy库
import tkinter as tk  # 用于创建图形用户界面的Tkinter库
from tkinter import filedialog, ttk, messagebox  # Tkinter的文件对话框、主题控件和消息框组件
from PIL import Image, ImageTk  # 用于图像处理和Tkinter图像转换的PIL库

# 统一图像尺寸配置（保留长方形比例，不压缩为正方形）
TARGET_WIDTH = 1500  # 目标图像的目标宽度
TARGET_HEIGHT = 880  # 目标图像的目标高度


class Rect:
    """矩形处理工具类，提供矩形坐标处理相关的静态方法"""
    
    @staticmethod
    def rectify(approx):
        """
        规整四边形坐标，按左上、右上、右下、左下的顺序排序
        
        参数:
            approx: 原始四边形坐标数组，形状为(4, 2)
            
        返回:
            rect: 排序后的后的坐标数组，按指定顺序排列
        """
        approx = approx.reshape(4, 2)  # 重塑为4行2列的数组
        rect = np.zeros((4, 2), dtype="float32")  # 初始化结果数组
        
        # 按坐标和排序确定左上和右下点
        s = approx.sum(axis=1)  # 计算每个点的x+y之和
        rect[0] = approx[np.argmin(s)]  # 最小和的点为左上
        rect[2] = approx[np.argmax(s)]  # 最大和的点为右下
        
        # 按坐标差排序确定右上和左下点
        diff = np.diff(approx, axis=1)  # 计算每个点的x-y差值
        rect[1] = approx[np.argmin(diff)]  # 最小差值的点为右上
        rect[3] = approx[np.argmax(diff)]  # 最大差值的点为左下
        
        return rect


class ScanApp:
    """文档扫描应用主类，负责UI创建、图像加载与处理的核心逻辑"""
    
    def __init__(self, root):
        """
        初始化扫描应用
        
        参数:
            root: Tkinter的主窗口对象
        """
        self.root = root  # 主窗口引用
        self.root.title("文档扫描应用")  # 设置窗口标题
        self.root.geometry("1400x900")  # 设置窗口初始大小
        
        self.image_path = None  # 图像文件路径
        self.original_image = None  # 原始图像数据
        self.processed_images = {}  # 存储处理后的各种图像
        self.zoom_window = None  # 图像放大窗口引用
        
        self.create_widgets()  # 创建UI组件
        
    def create_widgets(self):
        """创建应用程序的所有UI组件，包括控制区和图像显示区"""
        # 顶部控制区
        control_frame = ttk.Frame(self.root, padding=10)  # 创建控制框架
        control_frame.pack(fill=tk.X)  # 水平填充
        
        # 添加选择图像按钮
        ttk.Button(control_frame, text="选择图像", command=self.load_image).pack(side=tk.LEFT, padx=5)
        # 添加处理图像按钮
        ttk.Button(control_frame, text="处理图像", command=self.process_image).pack(side=tk.LEFT, padx=5)
        
        # 图像显示区
        self.display_frame = ttk.Frame(self.root, padding=10)  # 创建显示框架
        self.display_frame.pack(fill=tk.BOTH, expand=True)  # 填充并可扩展
        
        # 创建各个图像显示区域
        self.image_labels = {
            "原始图像": self.create_image_display_area("原始图像", 0, 0),
            "灰度图像": self.create_image_display_area("灰度图像", 0, 1),
            "模糊图像": self.create_image_display_area("模糊图像", 0, 2),
            "边缘检测": self.create_image_display_area("边缘检测", 1, 0),
            "轮廓标记": self.create_image_display_area("轮廓标记", 1, 1),
            "二值阈值": self.create_image_display_area("二值阈值", 1, 2),
            "均值阈值": self.create_image_display_area("均值阈值", 2, 0),
            "高斯阈值": self.create_image_display_area("高斯阈值", 2, 1),
            "Otsu阈值": self.create_image_display_area("Otsu阈值", 2, 2),
            "透视变换": self.create_image_display_area("透视变换", 3, 0)
        }
        
        # 状态标签
        self.status_label = ttk.Label(self.root, text="请选择一张图像开始操作", anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # 配置网格权重，使图像区域可随窗口大小调整
        for row in range(4):
            self.display_frame.grid_rowconfigure(row, weight=1)
        for col in range(3):
            self.display_frame.grid_columnconfigure(col, weight=1)
        
    def create_image_display_area(self, name, row, col):
        """
        创建图像显示区域，支持点击放大和右键保存功能
        
        参数:
            name: 图像区域的名称
            row: 网格布局中的行索引
            col: 网格布局中的列索引
            
        返回:
            canvas: 创建的画布对象，用于显示图像
        """
        frame = ttk.Frame(self.display_frame)  # 创建图像显示框架
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")  # 放置在网格中
        
        ttk.Label(frame, text=name).pack(anchor=tk.NW)  # 添加名称标签
        # 创建画布用于显示图像
        canvas = tk.Canvas(frame, width=300, height=200, bg="gray", cursor="hand2")
        canvas.pack(fill=tk.BOTH, expand=True)  # 填充并可扩展
        
        # 绑定左键点击事件用于放大图像
        canvas.bind("<Button-1>", lambda e, n=name: self.zoom_image(n))
        # 绑定右键点击事件用于显示保存菜单
        canvas.bind("<Button-3>", lambda e, n=name: self.show_save_menu(e, n))
        
        return canvas
    
    def show_save_menu(self, event, image_name):
        """
        显示右键保存菜单
        
        参数:
            event: 鼠标事件对象
            image_name: 图像名称，用于标识要保存的图像
        """
        if image_name not in self.processed_images:  # 检查图像是否存在
            return
        # 创建右键菜单
        menu = tk.Menu(self.root, tearoff=0)
        # 添加保存图像命令
        menu.add_command(label="保存图像", command=lambda: self.save_image(image_name))
        # 在鼠标位置显示菜单
        menu.post(event.x_root, event.y_root)
    
    def load_image(self):
        """
        加载图像文件（修复中文路径读取问题）
        使用filedialog选择文件，通过numpy和OpenCV的imdecode处理中文路径
        """
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:  # 如果选择了文件
            self.image_path = file_path
            try:
                # 修复OpenCV无法读取中文路径的问题
                # np.fromfile: 从文件读取数据到数组，支持中文路径
                img_array = np.fromfile(file_path, dtype=np.uint8)
                # cv2.imdecode: 从内存缓存中读取图像，第二个参数为读取模式
                self.original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if self.original_image is None:  # 检查图像是否读取成功
                    raise Exception("图像文件损坏或格式不支持")
                    
            except Exception as e:  # 处理异常
                messagebox.showerror("读取失败", f"无法加载图像：{str(e)}\n请检查文件路径和格式")
                self.original_image = None
                return

            # 更新状态标签
            self.status_label.config(text=f"已成功加载图像: {file_path}")
            self.processed_images.clear()  # 清空之前处理的图像
            
            # 统一原始图像尺寸
            unified_original = self.unify_image_size(self.original_image, TARGET_WIDTH, TARGET_HEIGHT, maintain_aspect_ratio=True)
            self.processed_images["原始图像"] = unified_original.copy()
            
            # 显示原始图像
            self.display_image(self.processed_images["原始图像"], self.image_labels["原始图像"])
            
            # 清空其他图像区域
            other_images = [name for name in self.image_labels if name != "原始图像"]
            for name in other_images:
                self.image_labels[name].delete("all")
                if name in self.processed_images:
                    del self.processed_images[name]
    
    def unify_image_size(self, cv_image, target_width, target_height, maintain_aspect_ratio=True):
        """
        统一图像尺寸，可选择保持纵横比
        
        参数:
            cv_image: OpenCV格式的图像
            target_width: 目标宽度
            target_height: 目标高度
            maintain_aspect_ratio: 是否保持纵横比，默认为True
            
        返回:
            调整尺寸后的图像
        """
        if cv_image is None:  # 检查输入图像是否有效
            raise ValueError("输入图像为None，请检查图像是否正确加载")
            
        if maintain_aspect_ratio:  # 保持纵横比的情况
            h, w = cv_image.shape[:2]  # 获取图像高度和宽度
            # 计算缩放比例，取宽度和高度比例的最小值
            ratio = min(target_width / w, target_height / h)
            new_size = (int(w * ratio), int(h * ratio))  # 计算新尺寸
            # cv2.resize: 调整图像大小，INTER_AREA适合缩小图像
            resized_image = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_AREA)
            
            # 创建目标画布并居中放置图像
            if len(cv_image.shape) == 3:  # 彩色图像
                target_canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            else:  # 灰度图像
                target_canvas = np.zeros((target_height, target_width), dtype=np.uint8)
                
            # 计算放置图像的起始坐标（居中）
            start_x = (target_width - new_size[0]) // 2
            start_y = (target_height - new_size[1]) // 2
            
            # 将调整后的图像放置到画布中央
            if len(cv_image.shape) == 3:
                target_canvas[start_y:start_y+new_size[1], start_x:start_x+new_size[0], :] = resized_image
            else:
                target_canvas[start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = resized_image
                
            return target_canvas
        else:  # 不保持纵横比，直接调整到目标尺寸
            return cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def display_image(self, cv_image, canvas):
        """
        在Tkinter画布上显示OpenCV格式的图像
        
        参数:
            cv_image: OpenCV格式的图像
            canvas: 要显示图像的Tkinter画布
        """
        # 获取画布尺寸，设置默认值防止初始为0
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 300
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 200
        
        h, w = cv_image.shape[:2]  # 获取图像尺寸
        # 计算缩放比例以适应画布
        ratio = min(canvas_width / w, canvas_height / h)
        new_size = (int(w * ratio), int(h * ratio))  # 计算新尺寸
        # 调整图像大小
        resized = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_AREA)
        
        # 转换颜色空间并适配Tkinter
        if len(cv_image.shape) == 3:  # 彩色图像：BGR转RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:  # 灰度图像：转RGB以便显示
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
        # 转换为PIL图像再转为Tkinter可用的图像格式
        pil_image = Image.fromarray(rgb_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # 在画布上显示图像
        canvas.delete("all")  # 清空画布
        # 居中显示图像
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=tk_image)
        canvas.image = tk_image  # 保留引用防止垃圾回收
    
    def zoom_image(self, image_name):
        """
        放大显示指定名称的图像
        
        参数:
            image_name: 要放大显示的图像名称
        """
        if image_name not in self.processed_images:  # 检查图像是否存在
            messagebox.showinfo("提示", "请先处理图像后再进行放大操作")
            return
            
        # 关闭已有放大窗口
        if self.zoom_window and self.zoom_window.winfo_exists():
            self.zoom_window.destroy()
            
        # 创建新窗口用于放大显示
        self.zoom_window = tk.Toplevel(self.root)
        self.zoom_window.title(f"放大查看：{image_name}")  # 设置窗口标题
        
        # 设置窗口大小为屏幕尺寸减去边距
        screen_width = self.root.winfo_screenwidth() - 100
        screen_height = self.root.winfo_screenheight() - 100
        self.zoom_window.geometry(f"{screen_width}x{screen_height}")
        
        # 加载并转换图像
        original_image = self.processed_images[image_name]
        if len(original_image.shape) == 3:  # 彩色图像转RGB
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:  # 灰度图像转RGB
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            
        pil_image = Image.fromarray(rgb_image)  # 转为PIL图像
        image_width, image_height = pil_image.size  # 获取图像尺寸
        
        # 计算缩放比例以适应窗口
        scale_ratio = min(screen_width / image_width, screen_height / image_height)
        if scale_ratio < 1:  # 图像大于窗口时缩小
            new_width = int(image_width * scale_ratio)
            new_height = int(image_height * scale_ratio)
            # 高质量缩放图像
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        # 显示图像
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas = tk.Canvas(self.zoom_window, bg="white")  # 创建画布
        canvas.pack(fill=tk.BOTH, expand=True)  # 填充窗口
        
        # 获取画布尺寸
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else screen_width
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else screen_height
        # 计算居中显示的坐标
        image_x = (canvas_width - pil_image.size[0]) // 2
        image_y = (canvas_height - pil_image.size[1]) // 2
        
        # 在画布上放置图像
        canvas.create_image(image_x, image_y, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image  # 保留引用
        
        # 添加滚动条
        scrollbar_x = ttk.Scrollbar(canvas, orient=tk.HORIZONTAL, command=canvas.xview)
        scrollbar_y = ttk.Scrollbar(canvas, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
        
        # 放置滚动条
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.config(scrollregion=canvas.bbox("all"))  # 设置滚动区域
    
    def save_image(self, image_name):
        """
        保存指定名称的图像（支持中文路径）
        
        参数:
            image_name: 要保存的图像名称
        """
        if image_name not in self.processed_images:  # 检查图像是否存在
            messagebox.showinfo("提示", "请先处理图像后再进行保存操作")
            return
            
        image = self.processed_images[image_name]  # 获取图像数据
        default_filename = f"{image_name}.jpg"  # 默认文件名
        
        # 打开保存文件对话框
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG图像文件", "*.jpg"), ("PNG图像文件", "*.png"), ("所有文件", "*.*")],
            initialfile=default_filename
        )
        
        if save_path:  # 如果选择了保存路径
            try:
                # 支持中文路径保存
                ext = save_path.split('.')[-1].lower()  # 获取文件扩展名
                # 设置编码参数
                if ext == 'png':
                    encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # PNG压缩级别
                else:
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]  # JPEG质量
                
                # cv2.imencode: 将图像编码为指定格式的内存缓存
                retval, im_buf_arr = cv2.imencode(f'.{ext}', image, encode_param)
                im_buf_arr.tofile(save_path)  # 保存到文件，支持中文路径
                
                messagebox.showinfo("成功", f"图像已成功保存至：\n{save_path}")
            except Exception as e:  # 处理保存异常
                messagebox.showerror("错误", f"图像保存失败：\n{str(e)}")
    
    def process_image(self):
        """
        处理图像的核心逻辑，包括灰度化、模糊、边缘检测、轮廓识别、透视变换和阈值处理
        """
        if self.original_image is None:  # 检查是否已加载图像
            self.status_label.config(text="请先加载一张图像再进行处理")
            messagebox.showwarning("警告", "请先选择并加载一张图像后再进行处理操作！")
            return
            
        try:
            # 1. 预处理原始图像
            image = self.unify_image_size(self.original_image, TARGET_WIDTH, TARGET_HEIGHT, maintain_aspect_ratio=True)
            original = image.copy()  # 保存原始图像副本
            self.processed_images["原始图像"] = original.copy()  # 存储原始图像
            
            # 2. 灰度化和模糊处理
            # cv2.cvtColor: 转换颜色空间，BGR2GRAY将彩色图转为灰度图
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.GaussianBlur: 高斯模糊，减少噪声，参数为卷积核大小和标准差
            blurred = cv2.GaussianBlur(grayscale, (7, 7), 1.5)
            # 存储处理后的图像
            self.processed_images["灰度图像"] = self.unify_image_size(grayscale, TARGET_WIDTH, TARGET_HEIGHT, True)
            self.processed_images["模糊图像"] = self.unify_image_size(blurred, TARGET_WIDTH, TARGET_HEIGHT, True)
            
            # 3. 边缘检测
            # cv2.Canny: Canny边缘检测算法，参数为低阈值和高阈值
            edges = cv2.Canny(blurred, threshold1=30, threshold2=150)
            # 创建矩形结构元素用于形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # cv2.morphologyEx: 形态学操作，MORPH_CLOSE用于闭合边缘中的小缺口
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            self.processed_images["边缘检测"] = self.unify_image_size(edges, TARGET_WIDTH, TARGET_HEIGHT, True)
            
            # 4. 轮廓检测
            # cv2.findContours: 查找图像中的轮廓，RETR_EXTERNAL只检测外轮廓，CHAIN_APPROX_SIMPLE压缩水平、垂直和对角线方向的轮廓点
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 按轮廓面积降序排序，取前20个较大的轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
            
            target_contour = None  # 目标轮廓（文档的四边形轮廓）
            for c in contours:
                # cv2.arcLength: 计算轮廓周长，True表示轮廓是闭合的
                perimeter = cv2.arcLength(c, True)
                # cv2.approxPolyDP: 多边形逼近，第二个参数为逼近精度（周长的百分比）
                approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                # 寻找面积足够大的四边形轮廓
                if len(approx) == 4 and cv2.contourArea(c) > 5000:
                    target_contour = approx
                    break
            
            # 5. 轮廓标记
            contour_marking = original.copy()  # 复制原始图像用于绘制轮廓
            if target_contour is not None:  # 如果找到目标轮廓
                # cv2.drawContours: 绘制轮廓，-1表示绘制所有轮廓，颜色为绿色(0,255,0)，线宽5
                cv2.drawContours(contour_marking, [target_contour], -1, (0, 255, 0), 5)
                self.status_label.config(text="已找到文档轮廓，处理完成（保留原轮廓长宽比）")
            else:  # 未找到有效轮廓
                # cv2.putText: 在图像上绘制文字，参数包括文字内容、位置、字体、大小、颜色和线宽
                cv2.putText(contour_marking, "未找到有效四边形轮廓", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                self.status_label.config(text="未找到有效文档轮廓，请尝试使用更清晰的图像")
                
            self.processed_images["轮廓标记"] = self.unify_image_size(contour_marking, TARGET_WIDTH, TARGET_HEIGHT, True)
            
            # 6. 透视变换 - 核心改动：保持原轮廓长宽比
            # 创建默认占位图（灰色）
            perspective_result = np.zeros((500, 500, 3), dtype=np.uint8) + 127
            empty_threshold = np.zeros((500, 500), dtype=np.uint8) + 127  # 阈值处理的占位图
            
            if target_contour is not None:  # 如果找到目标轮廓
                # 规整轮廓坐标
                approx = Rect.rectify(target_contour)
                
                # 计算原始轮廓的宽度和高度
                # np.linalg.norm: 计算向量的欧氏距离
                width1 = np.linalg.norm(approx[0] - approx[1])  # 上边长
                width2 = np.linalg.norm(approx[2] - approx[3])  # 下边长
                height1 = np.linalg.norm(approx[1] - approx[2])  # 右边长
                height2 = np.linalg.norm(approx[3] - approx[0])  # 左边长
                
                # 取平均宽度和高度
                width = (width1 + width2) / 2
                height = (height1 + height2) / 2
                
                # 保持原始长宽比，设置目标宽度为1200，高度按比例计算
                target_width = 1200
                target_height = int(target_width * (height / width))
                
                # 确保高度合理
                if target_height < 100:
                    target_height = 100
                    
                # 透视变换
                # 目标点坐标（左上、右上、右下、左下）
                pts2 = np.float32([[0,0], [target_width,0], [target_width,target_height], [0,target_height]])
                # cv2.getPerspectiveTransform: 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(approx, pts2)
                # cv2.warpPerspective: 应用透视变换
                perspective_result = cv2.warpPerspective(original, M, (target_width, target_height))
                
            self.processed_images["透视变换"] = self.unify_image_size(perspective_result, TARGET_WIDTH, TARGET_HEIGHT, True)
            
            # 7. 阈值处理（文档二值化）
            # 检查透视变换结果是否有效
            if len(perspective_result.shape) == 3 and not np.all(perspective_result == 127):
                # 转为灰度图
                perspective_gray = cv2.cvtColor(perspective_result, cv2.COLOR_BGR2GRAY)
                # 统一阈值图像尺寸以匹配显示区域
                perspective_gray = self.unify_image_size(perspective_gray, TARGET_WIDTH, TARGET_HEIGHT, True)
                
                # 各种阈值计算
                # cv2.threshold: 简单阈值处理，127为阈值，255为最大值，THRESH_BINARY为二值化模式
                ret, binary = cv2.threshold(perspective_gray, 127, 255, cv2.THRESH_BINARY)
                # cv2.adaptiveThreshold: 自适应阈值，根据局部区域计算阈值，MEAN_C使用均值
                mean = cv2.adaptiveThreshold(perspective_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
                # GAUSSIAN_C使用高斯加权均值
                gaussian = cv2.adaptiveThreshold(perspective_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                # THRESH_OTSU自动计算最佳阈值
                ret2, otsu = cv2.threshold(perspective_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 存储各种阈值处理结果
                self.processed_images["二值阈值"] = self.unify_image_size(binary, TARGET_WIDTH, TARGET_HEIGHT, True)
                self.processed_images["均值阈值"] = self.unify_image_size(mean, TARGET_WIDTH, TARGET_HEIGHT, True)
                self.processed_images["高斯阈值"] = self.unify_image_size(gaussian, TARGET_WIDTH, TARGET_HEIGHT, True)
                self.processed_images["Otsu阈值"] = self.unify_image_size(otsu, TARGET_WIDTH, TARGET_HEIGHT, True)
            else:
                # 无有效轮廓时填充占位图
                for key in ["二值阈值", "均值阈值", "高斯阈值", "Otsu阈值"]:
                    self.processed_images[key] = empty_threshold.copy()
            
            # 8. 显示所有处理结果
            for name, img in self.processed_images.items():
                self.display_image(img, self.image_labels[name])
                
        except Exception as e:  # 处理图像处理过程中的异常
            self.status_label.config(text=f"处理出错：{str(e)}")
            messagebox.showerror("处理错误", f"图像处理过程中出现异常：\n{str(e)}")
            print(f"详细错误信息：{e}")


if __name__ == "__main__":
    root = tk.Tk()  # 创建Tkinter主窗口
    app = ScanApp(root)  # 实例化扫描应用
    root.mainloop()  # 启动主事件循环
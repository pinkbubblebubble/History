import requests
from serpapi import GoogleSearch
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from smolagents import Tool

class GoogleLensManager:
    """A simple manager for using Google Lens search functionality"""

    def __init__(
        self,
        imgbb_api_key: str,
        serpapi_api_key: str,
        downloads_folder: Optional[str] = "downloads"
    ):
        self.imgbb_api_key = imgbb_api_key
        self.serpapi_api_key = serpapi_api_key
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()  # Record search history
        self.current_results: Dict = {}  # Current search results
        
        # Ensure the downloads folder exists
        os.makedirs(downloads_folder, exist_ok=True)

    def _upload_to_imgbb(self, image_path: str) -> Optional[str]:
        """Upload an image to ImgBB and return the URL"""
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: File does not exist - {image_path}")
            return None
            
        try:
            # Check if the file is readable
            if not os.access(image_path, os.R_OK):
                print(f"Error: File cannot be accessed - {image_path}")
                return None
                
            with open(image_path, "rb") as f:
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    params={"key": self.imgbb_api_key},
                    files={"image": f}
                )
            if response.status_code == 200:
                return response.json()["data"]["url"]
            else:
                print(f"Upload failed, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error occurred while uploading the image: {str(e)}")
            return None

    def _google_lens_search(self, image_url: str) -> Dict:
        """Perform a Google Lens search using the image"""
        if not self.serpapi_api_key:
            print("Error: SERPAPI_API_KEY not provided")
            return {"error": "Missing SERPAPI_API_KEY"}
            
        params = {
            "engine": "google_lens",
            "url": image_url,
            "hl": "en",
            "country": "us",
            "api_key": self.serpapi_api_key
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            # Record search history
            self.history.append((image_url, time.time()))
            self.current_results = results
            return results
        except Exception as e:
            print(f"Error occurred during Google Lens search: {str(e)}")
            return {"error": str(e)}

    def get_visual_matches(self, limit: int = 5) -> List[Dict]:
        """Get visual match results"""
        if "visual_matches" in self.current_results:
            return self.current_results["visual_matches"][:limit]
        return []

    def get_related_content(self, limit: int = 5) -> List[Dict]:
        """Get related content"""
        if "related_content" in self.current_results:
            return self.current_results["related_content"][:limit]
        return []

    def get_urls(self, limit: int = 5) -> List[str]:
        """Get all related URLs"""
        urls = []
        
        # Get URLs from visual matches
        for match in self.get_visual_matches(limit):
            if "link" in match:
                urls.append(match["link"])
                
        # Get URLs from related content
        for content in self.get_related_content(limit):
            if "link" in content:
                urls.append(content["link"])
                
        return list(set(urls))[:limit]  # Remove duplicates and limit the number

    def search(self, image_path: str) -> Dict:
        """Perform an image search and return the results"""
        # Normalize the path
        image_path = os.path.normpath(image_path)
        
        # Check if the file exists
        if not os.path.exists(image_path):
            error_msg = f"File does not exist: {image_path}"
            print(error_msg)
            return {"error": error_msg}
            
        # Check if the file is readable
        if not os.access(image_path, os.R_OK):
            error_msg = f"File cannot be accessed: {image_path}"
            print(error_msg)
            return {"error": error_msg}
            
        # Check the file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            error_msg = f"Unsupported file format: {ext}"
            print(error_msg)
            return {"error": error_msg}
            
        # Upload the image
        image_url = self._upload_to_imgbb(image_path)
        if not image_url:
            error_msg = "Image upload failed"
            print(error_msg)
            return {"error": error_msg}
            
        # Search the image
        results = self._google_lens_search(image_url)
        if "error" in results:
            return results
            
        return {
            "image_url": image_url,
            "visual_matches": self.get_visual_matches(),
            "related_content": self.get_related_content(),
            "urls": self.get_urls()
        }

    @property
    def last_search(self) -> Optional[Tuple[str, float]]:
        """Get the information of the most recent search"""
        return self.history[-1] if self.history else None
    
class GoogleLensSearchTool(Tool):
    name = "google_lens_search"
    description = """Use Google Lens to search for images and return related web URLs and visual match results.
    Input the image file path, and return:
    1. Online URL of the image
    2. Visual match results
    3. Related web links
    Note: The image must be a valid image file (.jpg, .png, .jpeg, etc.)"""
    
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to search, note the relative path",
            "nullable": True
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 5,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, imgbb_api_key: str, serpapi_api_key: str):
        super().__init__()
        self.manager = GoogleLensManager(imgbb_api_key, serpapi_api_key)

    def forward(self, image_path: str = None, limit: int = 5) -> str:
        """Perform the search and return formatted results"""
        if not image_path:
            return "Error: Image path not provided"
            
        # Normalize the path
        try:
            image_path = os.path.normpath(image_path)
        except Exception as e:
            return f"Error: Invalid path format - {str(e)}"
            
        # Validate the file format
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            return f"Error: Unsupported image format {ext}"
            
        # Perform the search
        try:
            results = self.manager.search(image_path)
        except Exception as e:
            return f"Error: An exception occurred during the search - {str(e)}"
            
        if "error" in results:
            return f"Error: {results['error']}"
            
        # Format the output
        output = [
            f"# Google Lens Search Results",
            f"Original Image Path: {image_path}",
            f"Online Image URL: {results['image_url']}",
            "\n## Search Results:"
        ]
        
        # Combine visual matches and related content
        all_matches = []
        if results.get('visual_matches'):
            all_matches.extend(results['visual_matches'])
        if results.get('related_content'):
            all_matches.extend(results['related_content'])
            
        # Limit the number of results and format each match
        for i, match in enumerate(all_matches[:limit], 1):
            output.extend([
                f"\n### Result {i}:",
                f"- Title: {match.get('title', 'Unknown title')}",
                f"- Source: {match.get('source', 'Unknown source')}",
                f"- Link: {match.get('link', 'No link')}",
            ])
            
            # Add thumbnail and original image information if available
            if match.get('thumbnail'):
                output.append(f"- Thumbnail: {match['thumbnail']}")
            if match.get('image'):
                output.append(f"- Original Image: {match['image']}")
            
            # Add other possible information
            for key, value in match.items():
                if key not in ['title', 'source', 'link', 'thumbnail', 'image']:
                    output.append(f"- {key}: {value}")
                
        return "\n".join(output)
# import requests
# from serpapi import GoogleSearch
# import json
# import os
# import time
# from typing import Dict, Any, List, Optional, Tuple
# from smolagents import Tool

# class GoogleLensManager:
#     """管理Google Lens搜索功能的简单管理器"""

#     def __init__(
#         self,
#         imgbb_api_key: str,
#         serpapi_api_key: str,
#         downloads_folder: Optional[str] = "downloads"
#     ):
#         self.imgbb_api_key = imgbb_api_key
#         self.serpapi_api_key = serpapi_api_key
#         self.downloads_folder = downloads_folder
#         self.history: List[Tuple[str, float]] = list()  # 记录搜索历史
#         self.current_results: Dict = {}  # 当前搜索结果
        
#         # 确保下载文件夹存在
#         os.makedirs(downloads_folder, exist_ok=True)

#     def _upload_to_imgbb(self, image_path: str) -> Optional[str]:
#         """上传图片到ImgBB并返回URL"""
#         # 添加文件存在性检查
#         if not os.path.exists(image_path):
#             print(f"错误：文件不存在 - {image_path}")
#             return None
            
#         try:
#             # 检查文件是否可读
#             if not os.access(image_path, os.R_OK):
#                 print(f"错误：文件无法访问 - {image_path}")
#                 return None
                
#             with open(image_path, "rb") as f:
#                 response = requests.post(
#                     "https://api.imgbb.com/1/upload",
#                     params={"key": self.imgbb_api_key},
#                     files={"image": f}
#                 )
#             if response.status_code == 200:
#                 return response.json()["data"]["url"]
#             else:
#                 print(f"上传失败，状态码：{response.status_code}")
#                 return None
#         except Exception as e:
#             print(f"上传图片时出错: {str(e)}")
#             return None

#     def _google_lens_search(self, image_url: str) -> Dict:
#         """使用Google Lens搜索图片"""
#         if not self.serpapi_api_key:
#             print("错误：未提供 SERPAPI_API_KEY")
#             return {"error": "Missing SERPAPI_API_KEY"}
            
#         params = {
#             "engine": "google_lens",
#             "url": image_url,
#             "hl": "en",
#             "country": "us",
#             "api_key": self.serpapi_api_key
#         }
        
#         try:
#             search = GoogleSearch(params)
#             results = search.get_dict()
#             # 记录搜索历史
#             self.history.append((image_url, time.time()))
#             self.current_results = results
#             return results
#         except Exception as e:
#             print(f"Google Lens搜索出错: {str(e)}")
#             return {"error": str(e)}

#     def get_visual_matches(self, limit: int = 5) -> List[Dict]:
#         """获取视觉匹配结果"""
#         if "visual_matches" in self.current_results:
#             return self.current_results["visual_matches"][:limit]
#         return []

#     def get_related_content(self, limit: int = 5) -> List[Dict]:
#         """获取相关内容"""
#         if "related_content" in self.current_results:
#             return self.current_results["related_content"][:limit]
#         return []

#     def get_urls(self, limit: int = 5) -> List[str]:
#         """获取所有相关URL"""
#         urls = []
        
#         # 从视觉匹配中获取URL
#         for match in self.get_visual_matches(limit):
#             if "link" in match:
#                 urls.append(match["link"])
                
#         # 从相关内容中获取URL
#         for content in self.get_related_content(limit):
#             if "link" in content:
#                 urls.append(content["link"])
                
#         return list(set(urls))[:limit]  # 去重并限制数量

#     def search(self, image_path: str) -> Dict:
#         """执行图片搜索并返回结果"""
#         # 规范化路径
#         image_path = os.path.normpath(image_path)
        
#         # 检查文件是否存在
#         if not os.path.exists(image_path):
#             error_msg = f"文件不存在: {image_path}"
#             print(error_msg)
#             return {"error": error_msg}
            
#         # 检查文件是否可读
#         if not os.access(image_path, os.R_OK):
#             error_msg = f"文件无法访问: {image_path}"
#             print(error_msg)
#             return {"error": error_msg}
            
#         # 检查文件扩展名
#         ext = os.path.splitext(image_path)[1].lower()
#         if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
#             error_msg = f"不支持的文件格式: {ext}"
#             print(error_msg)
#             return {"error": error_msg}
            
#         # 上传图片
#         image_url = self._upload_to_imgbb(image_path)
#         if not image_url:
#             error_msg = "图片上传失败"
#             print(error_msg)
#             return {"error": error_msg}
            
#         # 搜索图片
#         results = self._google_lens_search(image_url)
#         if "error" in results:
#             return results
            
#         return {
#             "image_url": image_url,
#             "visual_matches": self.get_visual_matches(),
#             "related_content": self.get_related_content(),
#             "urls": self.get_urls()
#         }

#     @property
#     def last_search(self) -> Optional[Tuple[str, float]]:
#         """获取最近一次搜索的信息"""
#         return self.history[-1] if self.history else None
    
# class GoogleLensSearchTool(Tool):
#     name = "google_lens_search"
#     description = """使用Google Lens搜索图片并返回相关的网页URL和视觉匹配结果。
#     输入图片文件路径，返回:
#     1. 图片的在线URL
#     2. 视觉匹配结果
#     3. 相关网页链接
#     注意: 图片必须是有效的图片文件(.jpg, .png, .jpeg等)"""
    
#     inputs = {
#         "image_path": {
#             "type": "string",
#             "description": "要搜索的图片文件路径,请注意相对路径",
#             "nullable": True
#         },
#         "limit": {
#             "type": "integer",
#             "description": "返回结果的最大数量",
#             "default": 5,
#             "nullable": True
#         }
#     }
#     output_type = "string"

#     def __init__(self, imgbb_api_key: str, serpapi_api_key: str):
#         super().__init__()
#         self.manager = GoogleLensManager(imgbb_api_key, serpapi_api_key)

#     def forward(self, image_path: str = None, limit: int = 5) -> str:
#         """执行搜索并返回格式化的结果"""
#         if not image_path:
#             return "错误：未提供图片路径"
            
#         # 规范化路径
#         try:
#             image_path = os.path.normpath(image_path)
#         except Exception as e:
#             return f"错误：路径格式无效 - {str(e)}"
            
#         # 验证文件格式
#         ext = os.path.splitext(image_path)[1].lower()
#         if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
#             return f"错误：不支持的图片格式 {ext}"
            
#         # 执行搜索
#         try:
#             results = self.manager.search(image_path)
#         except Exception as e:
#             return f"错误：搜索过程中出现异常 - {str(e)}"
            
#         if "error" in results:
#             return f"错误：{results['error']}"
            
#         # 重新格式化输出
#         output = [
#             f"# Google Lens 搜索结果",
#             f"原始图片路径: {image_path}",
#             f"在线图片URL: {results['image_url']}",
#             "\n## 搜索结果:"
#         ]
        
#         # 合并视觉匹配和相关内容
#         all_matches = []
#         if results.get('visual_matches'):
#             all_matches.extend(results['visual_matches'])
#         if results.get('related_content'):
#             all_matches.extend(results['related_content'])
            
#         # 限制结果数量并格式化每个匹配项
#         for i, match in enumerate(all_matches[:limit], 1):
#             output.extend([
#                 f"\n### 结果 {i}:",
#                 f"- 标题: {match.get('title', '未知标题')}",
#                 f"- 来源: {match.get('source', '未知来源')}",
#                 f"- 链接: {match.get('link', '无链接')}",
#             ])
            
#             # 如果有缩略图和原图信息，也添加进去
#             if match.get('thumbnail'):
#                 output.append(f"- 缩略图: {match['thumbnail']}")
#             if match.get('image'):
#                 output.append(f"- 原图: {match['image']}")
            
#             # 添加其他可能的信息
#             for key, value in match.items():
#                 if key not in ['title', 'source', 'link', 'thumbnail', 'image']:
#                     output.append(f"- {key}: {value}")
                
#         return "\n".join(output)

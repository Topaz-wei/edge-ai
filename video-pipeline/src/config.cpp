#include "edge/video/config.hpp"
#include <opencv2/core.hpp>
#include <set>
#include <stdexcept>
#include <limits>
#include <fstream>
#include <regex>
#include <iterator>
namespace edge::video {
namespace {
// FileStorage converts integer scalars to int before clients see them. Guard the
// original tokens first so 2^32+1 cannot masquerade as a valid boolean or limit.
void checkIntegerLiterals(const std::string& yaml) {
    std::string unquoted = yaml;
    char quote = 0;
    bool comment = false;
    bool scalarStart = true;
    for (std::size_t i = 0; i < yaml.size(); ++i) {
        const char c = yaml[i];
        if (comment) {
            if (c == '\n') { comment = false; scalarStart = true; }
            else unquoted[i] = ' ';
        } else if (quote) {
            if (c == '\n' || c == '\r') throw std::runtime_error("Multiline quoted scalars are not supported");
            unquoted[i] = ' ';
            if (quote == '"' && c == '\\' && i + 1 < yaml.size()) {
                unquoted[++i] = ' ';
            } else if (c == quote) {
                if (quote == '\'' && i + 1 < yaml.size() && yaml[i + 1] == '\'')
                    unquoted[++i] = ' ';
                else quote = 0;
            }
        } else if (scalarStart && (c == '"' || c == '\'')) {
            quote = c; scalarStart = false; unquoted[i] = ' ';
        } else if (c == '#') {
            comment = true; unquoted[i] = ' ';
        } else if (c == '\n' || c == '\r' || c == ',' || c == '[' || c == '{' ||
                   (c == ':' && i + 1 < yaml.size() && std::isspace(static_cast<unsigned char>(yaml[i + 1])))) {
            scalarStart = true;
        } else if (!std::isspace(static_cast<unsigned char>(c)) && !(scalarStart && c == '-')) {
            scalarStart = false;
        }
    }
    const std::regex scalar(R"([+-]?(0[xX][0-9a-fA-F]+|[0-9]+))");
    const std::string delimiters = " \t\r\n:,[]{}";
    for (std::size_t begin = unquoted.find_first_not_of(delimiters);
         begin != std::string::npos;) {
        auto end = unquoted.find_first_of(delimiters, begin);
        auto token = unquoted.substr(begin, end - begin);
        if (std::regex_match(token, scalar)) {
            try {
                const int base = token.find_first_of("xX") == std::string::npos ? 10 : 16;
                const auto value = std::stoll(token, nullptr, base);
                if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
                    throw std::out_of_range("integer");
            } catch (const std::out_of_range&) {
                throw std::runtime_error("YAML integer outside signed 32-bit range: " + token);
            }
        }
        begin = end == std::string::npos ? end : unquoted.find_first_not_of(delimiters, end);
    }
}
void fields(const cv::FileNode& node, const std::set<std::string>& allowed) {
    if(!node.isMap()) throw std::runtime_error("Expected YAML mapping");
    std::set<std::string> seen;
    for(auto i=node.begin();i!=node.end();++i) {
        auto name=(*i).name();
        if(!allowed.count(name) || !seen.insert(name).second)
            throw std::runtime_error("Unknown or duplicate field: "+name);
    }
}
std::string str(const cv::FileNode& n,const char* key,bool required=false) {
    auto v=n[key];
    if(v.empty()) { if(required) throw std::runtime_error(std::string("Missing ")+key); return {}; }
    if(!v.isString() || std::string(v).empty()) throw std::runtime_error(std::string("Expected nonempty string: ")+key);
    return std::string(v);
}
int integer(const cv::FileNode& n,const char* key,int fallback,int low,int high=std::numeric_limits<int>::max()) {
    auto v=n[key]; if(v.empty()) return fallback;
    if(!v.isInt()) throw std::runtime_error(std::string("Expected integer: ")+key);
    int value=v;
    if(value<low || value>high) throw std::runtime_error(std::string("Out of range: ")+key);
    return value;
}
std::filesystem::path resolve(const std::filesystem::path& base,const std::string& text) {
    if(text.empty()) return {};
    auto p=std::filesystem::path(text);
    return (p.is_absolute()?p:base/p).lexically_normal();
}
}
VideoConfig loadConfig(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open config: " + path.string());
    const std::string yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    checkIntegerLiterals(yaml);
    cv::FileStorage file(yaml,cv::FileStorage::READ | cv::FileStorage::MEMORY);
    if(!file.isOpened()) throw std::runtime_error("Cannot open config: "+path.string());
    auto root=file.root();
    fields(root,{"schema_version","source","display","output","timing_csv","max_frames","capture_width","capture_height","capture_fps","capture_buffers","capture_fourcc","models"});
    VideoConfig c;
    if(root["schema_version"].empty()) throw std::runtime_error("Missing schema_version");
    c.schemaVersion=integer(root,"schema_version",1,1,1);
    auto base=std::filesystem::absolute(path).parent_path();
    c.source=str(root,"source",true); c.camera=cameraSource(c.source);
    if(digits(c.source)) c.cameraIndex=std::stoi(c.source);
    if(!c.camera && c.source.find("://")==std::string::npos) c.source=resolve(base,c.source).string();
    c.display=integer(root,"display",1,0,1);
    c.output=resolve(base,str(root,"output")).string();
    c.timingCsv=resolve(base,str(root,"timing_csv")).string();
    c.maxFrames=integer(root,"max_frames",0,1);
    c.captureWidth=integer(root,"capture_width",0,1); c.captureHeight=integer(root,"capture_height",0,1);
    c.captureFps=integer(root,"capture_fps",0,1); c.captureBuffers=integer(root,"capture_buffers",2,1);
    c.captureFourcc=str(root,"capture_fourcc");
    if((c.captureWidth==0)!=(c.captureHeight==0)) throw std::runtime_error("Capture width/height must be paired");
    if(!c.captureFourcc.empty() && c.captureFourcc.size()!=4) throw std::runtime_error("FOURCC must have four characters");
    if(!c.camera) for(auto key:{"capture_width","capture_height","capture_fps","capture_buffers","capture_fourcc"})
        if(!root[key].empty()) throw std::runtime_error("Capture settings require a camera");
    auto models=root["models"];
    if(!models.isSeq() || models.size()==0) throw std::runtime_error("models must be nonempty sequence");
    std::set<std::string> ids;
    for(auto i=models.begin();i!=models.end();++i) {
        auto n=*i; fields(n,{"id","type","engine","input_stage","graph"});
        ModelConfig m; m.id=str(n,"id",true); m.type=str(n,"type",true);
        if(m.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
            throw std::runtime_error("Model id must use letters, digits, underscores or hyphens");
        if(!ids.insert(m.id).second) throw std::runtime_error("Duplicate model id: "+m.id);
        m.engine=resolve(base,str(n,"engine",true)); m.inputStage=str(n,"input_stage");
        m.graph=integer(n,"graph",1,0,1); c.models.push_back(std::move(m));
    }
    return c;
}
}

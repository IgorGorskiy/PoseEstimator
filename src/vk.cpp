#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#define GLM_ENABLE_EXPERIMENTAL
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#define VK_CHECK(expr)                                                   \
    do {                                                                 \
        VkResult _r = (expr);                                            \
        if (_r != VK_SUCCESS)                                            \
            throw std::runtime_error(std::string(#expr " → VkResult=") + \
                                     std::to_string(static_cast<int>(_r))); \
    } while (0)


namespace VK {

struct GpuVertex { float x, y, z; };

unsigned int width = 1024;
unsigned int height = 1024;
float lineWidth = 1;

VkInstance instance;
VkPhysicalDevice physicalDevice;
VkDevice device;
uint32_t queueFamilyIndex;
VkQueue queue;
VkCommandPool commandPool;
struct FrameBufferAttachment {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
};
FrameBufferAttachment colorAttachment, depthAttachment;
VkRenderPass renderPass;
VkFramebuffer framebuffer;
VkPipelineLayout pipeLayout;
VkPipeline facePipeline;
VkPipeline edgePipeline;

VkBuffer       readBuf  {};   VkDeviceMemory readMem   {};
void*          readPtr  {};   // persistently mapped

VkDebugReportCallbackEXT debugReportCallback;
VkCommandBuffer command_buffer;

VkFence fence;

size_t faces_count = 0;
size_t edges_count = 0;

struct MemoryBuffer {
    VkBuffer buf;
    VkDeviceMemory mem;
};

uint32_t findMemType(uint32_t filter, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((filter & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    throw std::runtime_error("No suitable memory type found");
}

void allocBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags memFlags,
                                    VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bCI.size        = size;
    bCI.usage       = usage;
    bCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bCI, nullptr, &buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, memFlags);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
}

uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
    for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        typeBits >>= 1;
    }
    return 0;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objectType,
	uint64_t object,
	size_t location,
	int32_t messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData)
{
	printf("[VALIDATION]: %s - %s\n", pLayerPrefix, pMessage);
	return VK_FALSE;
}

VkShaderModule compile_shader(const std::string& file_path, shaderc_shader_kind kind) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << file_path << std::endl;
        return {};
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Set target to Vulkan 1.2 or higher for modern features
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    
    // Optional: Set optimization levels (e.g., performance, size, or none)
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    // options.SetOptimizationLevel(shaderc_optimization_level_zero);
    // options.SetGenerateDebugInfo();

    // Perform the compilation
    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
        source, kind, "shader_source", options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cout << "Shader compilation failed: " << result.GetErrorMessage() << std::endl;
        return {};
    }

    // Return the SPIR-V binary as a vector of uint32_t
    std::vector<uint32_t> bytecode{result.cbegin(), result.cend()};

    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = bytecode.size() * 4;
    smCI.pCode    = bytecode.data();
    VkShaderModule sm{};
    VK_CHECK(vkCreateShaderModule(device, &smCI, nullptr, &sm));
    return sm;
}

void init() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan headless";
    appInfo.pEngineName = "Vulkan headless";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;

    uint32_t layerCount = 0;
    const char* validationLayers[] = { "VK_LAYER_LUNARG_standard_validation" };
    layerCount = 1;
#if DEBUG
    // Check if layers are available
    uint32_t instanceLayerCount;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
    std::vector<VkLayerProperties> instanceLayers(instanceLayerCount);
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayers.data());

    bool layersAvailable = true;
    for (auto layerName : validationLayers) {
        bool layerAvailable = false;
        for (auto instanceLayer : instanceLayers) {
            if (strcmp(instanceLayer.layerName, layerName) == 0) {
                layerAvailable = true;
                break;
            }
        }
        if (!layerAvailable) {
            layersAvailable = false;
            break;
        }
    }

    if (layersAvailable) {
        instanceCreateInfo.ppEnabledLayerNames = validationLayers;
        const char *validationExt = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
        instanceCreateInfo.enabledLayerCount = layerCount;
        instanceCreateInfo.enabledExtensionCount = 1;
        instanceCreateInfo.ppEnabledExtensionNames = &validationExt;
    }
#endif
    VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

#if DEBUG
    if (layersAvailable) {
        VkDebugReportCallbackCreateInfoEXT debugReportCreateInfo = {};
        debugReportCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        debugReportCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
        debugReportCreateInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT)debugMessageCallback;

        // We have to explicitly load this function.
        PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
        assert(vkCreateDebugReportCallbackEXT);
        VK_CHECK(vkCreateDebugReportCallbackEXT(instance, &debugReportCreateInfo, nullptr, &debugReportCallback));
    }
#endif

    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
    physicalDevice = physicalDevices[0];

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    printf("GPU: %s\n", deviceProperties.deviceName);

    // Request a single graphics queue
    const float defaultQueuePriority(0.0f);
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
        if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamilyIndex = i;
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = i;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
            break;
        }
    }
    // Create logical device
    VkPhysicalDeviceFeatures enabledFeatures{};
    // wideLines нужна для lineWidth > 1.0; проверяем поддержку GPU
    {
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physicalDevice, &supported);
        if (supported.wideLines)
            enabledFeatures.wideLines = VK_TRUE;
        else
            fprintf(stderr, "[VK] wideLines не поддерживается GPU — lineWidth будет ограничена 1.0\n");
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.pEnabledFeatures  = &enabledFeatures;
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    // Get a graphics queue
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));

    VkCommandBufferAllocateInfo cbAI{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAI.commandPool        = commandPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbAI, &command_buffer));

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    // image color and depth
    {
        auto allocImage = [&](VkFormat fmt, VkImageUsageFlags usage,
                          VkImage& img, VkDeviceMemory& mem, VkImageView& view,
                          VkImageAspectFlags aspect) {
            VkImageCreateInfo imgCI{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imgCI.imageType   = VK_IMAGE_TYPE_2D;
            imgCI.format      = fmt;
            imgCI.extent      = {width, height, 1};
            imgCI.mipLevels   = 1;
            imgCI.arrayLayers = 1;
            imgCI.samples     = VK_SAMPLE_COUNT_1_BIT;
            imgCI.tiling      = VK_IMAGE_TILING_OPTIMAL;
            imgCI.usage       = usage;
            imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VK_CHECK(vkCreateImage(device, &imgCI, nullptr, &img));

            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(device, img, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize  = req.size;
            ai.memoryTypeIndex = findMemType(req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
            VK_CHECK(vkBindImageMemory(device, img, mem, 0));

            VkImageViewCreateInfo viewCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewCI.image    = img;
            viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCI.format   = fmt;
            viewCI.subresourceRange = {aspect, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device, &viewCI, nullptr, &view));
        };

        // Цветовое вложение: рендер + копирование на CPU
        allocImage(colorFormat,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                colorAttachment.image, colorAttachment.memory, colorAttachment.view, VK_IMAGE_ASPECT_COLOR_BIT);

        // Буфер глубины
        allocImage(depthFormat,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                depthAttachment.image, depthAttachment.memory, depthAttachment.view, VK_IMAGE_ASPECT_DEPTH_BIT);

        // Readback buffer: HOST_VISIBLE | HOST_COHERENT, persistently mapped
        const VkDeviceSize readSize = static_cast<VkDeviceSize>(width) * height * 4; // RGBA8
        allocBuffer(readSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    readBuf, readMem);
        VK_CHECK(vkMapMemory(device, readMem, 0, readSize, 0, &readPtr));
    }
    // render pass
    {
        VkAttachmentDescription attachments[2]{};

        // Цвет: очищаем в чёрный, сохраняем для последующего копирования
        attachments[0].format         = colorFormat;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        // Render pass сам переводит в TRANSFER_SRC_OPTIMAL
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        // Глубина: очищаем в 1.0, результат не нужен
        attachments[1].format         = depthFormat;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        // Зависимости синхронизации
        VkSubpassDependency deps[2]{};

        // [0] Внешнее → subpass 0: readback предыдущего кадра завершён до записи
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        // [1] Subpass 0 → внешнее: цветовая запись завершена до копирования
        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpCI.attachmentCount = 2;
        rpCI.pAttachments    = attachments;
        rpCI.subpassCount    = 1;
        rpCI.pSubpasses      = &subpass;
        rpCI.dependencyCount = 2;
        rpCI.pDependencies   = deps;
        VK_CHECK(vkCreateRenderPass(device, &rpCI, nullptr, &renderPass));
    }
    // frame buffer
    {
        VkImageView views[2] = {colorAttachment.view, depthAttachment.view};

        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = renderPass;
        fbCI.attachmentCount = 2;
        fbCI.pAttachments    = views;
        fbCI.width           = static_cast<uint32_t>(width);
        fbCI.height          = static_cast<uint32_t>(height);
        fbCI.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(device, &fbCI, nullptr, &framebuffer));
    }
    // pipeline 
    {
        std::vector<VkPushConstantRange> consts_range;
        
        consts_range.push_back(VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(float) * 4
        });

        consts_range.push_back(VkPushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 16,
            .size = sizeof(float) * 4 * 4 * 2
        });

        VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutCI.pushConstantRangeCount = consts_range.size();
        layoutCI.pPushConstantRanges    = consts_range.data();
        VK_CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipeLayout));

        VkShaderModule vertSM = compile_shader("shader.vert", shaderc_vertex_shader);
        VkShaderModule fragSM = compile_shader("shader.frag", shaderc_fragment_shader);

        // ── Общие состояния ───────────────────────────────────────────────────────
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertSM;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragSM;
        stages[1].pName  = "main";

        VkVertexInputBindingDescription vtxBinding{};
        vtxBinding.binding   = 0;
        vtxBinding.stride    = sizeof(GpuVertex);
        vtxBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription vtxAttr{};
        vtxAttr.binding  = 0;
        vtxAttr.location = 0;
        vtxAttr.format   = VK_FORMAT_R32G32B32_SFLOAT;
        vtxAttr.offset   = 0;

        VkPipelineVertexInputStateCreateInfo viCI{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        viCI.vertexBindingDescriptionCount   = 1;
        viCI.pVertexBindingDescriptions      = &vtxBinding;
        viCI.vertexAttributeDescriptionCount = 1;
        viCI.pVertexAttributeDescriptions    = &vtxAttr;

        VkViewport vp{0.f, 0.f,
                    static_cast<float>(width),
                    static_cast<float>(height),
                    0.f, 1.f};
        VkRect2D scissor{{0, 0},
                        {static_cast<uint32_t>(width),
                        static_cast<uint32_t>(height)}};

        VkPipelineViewportStateCreateInfo vpCI{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpCI.viewportCount = 1; vpCI.pViewports = &vp;
        vpCI.scissorCount  = 1; vpCI.pScissors  = &scissor;

        VkPipelineMultisampleStateCreateInfo msCI{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Цвет: без блендинга
        VkPipelineColorBlendAttachmentState cbAtt{};
        cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cbCI{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbCI.attachmentCount = 1;
        cbCI.pAttachments    = &cbAtt;

        // Общий шаблон описания
        VkGraphicsPipelineCreateInfo pCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pCI.stageCount          = 2;
        pCI.pStages             = stages;
        pCI.pVertexInputState   = &viCI;
        pCI.pViewportState      = &vpCI;
        pCI.pMultisampleState   = &msCI;
        pCI.pColorBlendState    = &cbCI;
        pCI.layout              = pipeLayout;
        pCI.renderPass          = renderPass;
        pCI.subpass             = 0;

        // ── Pipeline для граней: TRIANGLE_LIST, depth write ON ───────────────────
        {
            VkPipelineInputAssemblyStateCreateInfo iaCI{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rasCI{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasCI.cullMode    = VK_CULL_MODE_NONE;   // показываем оба борта грани
            rasCI.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasCI.lineWidth   = 1.0f;

            VkPipelineDepthStencilStateCreateInfo dsCI{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsCI.depthTestEnable  = VK_TRUE;
            dsCI.depthWriteEnable = VK_TRUE;
            dsCI.depthCompareOp   = VK_COMPARE_OP_LESS;
            // dsCI.depthCompareOp   = VK_COMPARE_OP_GREATER;

            pCI.pInputAssemblyState  = &iaCI;
            pCI.pRasterizationState  = &rasCI;
            pCI.pDepthStencilState   = &dsCI;
            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                                &pCI, nullptr, &facePipeline));
        }

        // ── Pipeline для рёбер: LINE_LIST, depth write ON ────────────────────────
        {
            VkPipelineInputAssemblyStateCreateInfo iaCI{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            iaCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

            // Толщина задаётся динамически через vkCmdSetLineWidth()
            // (требует wideLines для значений > 1.0)
            VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_LINE_WIDTH };
            VkPipelineDynamicStateCreateInfo dynCI{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynCI.dynamicStateCount = 1;
            dynCI.pDynamicStates    = dynStates;

            VkPipelineRasterizationStateCreateInfo rasCI{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasCI.cullMode    = VK_CULL_MODE_NONE;
            rasCI.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasCI.lineWidth   = 1.0f;  // статическое значение игнорируется при dynamic state

            VkPipelineDepthStencilStateCreateInfo dsCI{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsCI.depthTestEnable  = VK_TRUE;
            dsCI.depthWriteEnable = VK_TRUE;
            dsCI.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
            // dsCI.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;
            // Небольшой bias — рёбра, лежащие на поверхности граней, не теряются
            dsCI.minDepthBounds = 0.0f;
            dsCI.maxDepthBounds = 1.0f;

            // Постоянный depth bias для рёбер на поверхности
            rasCI.depthBiasEnable         = VK_FALSE;
            rasCI.depthBiasConstantFactor = -1.0f;  // немного ближе к камере
            rasCI.depthBiasSlopeFactor    = -1.0f;
            // rasCI.depthBiasConstantFactor = 1.0f;  // немного ближе к камере
            // rasCI.depthBiasSlopeFactor    = 1.0f;

            pCI.pInputAssemblyState  = &iaCI;
            pCI.pRasterizationState  = &rasCI;
            pCI.pDepthStencilState   = &dsCI;
            pCI.pDynamicState        = &dynCI;
            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                                &pCI, nullptr, &edgePipeline));
            pCI.pDynamicState = nullptr;  // сбросить, чтобы не влиять на другие pipeline
        }
    }
    // fence
    {
        VkFenceCreateInfo fenceCI{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device, &fenceCI, nullptr, &fence));
    }
}

MemoryBuffer face_buffer, edge_buffer;

void upload_faces(std::vector<float> faces) {
    allocBuffer(faces.size() * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            face_buffer.buf, face_buffer.mem
        );
    void* ptr = nullptr;
    VK_CHECK(vkMapMemory(device, face_buffer.mem, 0, faces.size() * sizeof(float), 0, &ptr));
    std::memcpy(ptr, faces.data(), faces.size() * sizeof(float));
    vkUnmapMemory(device, face_buffer.mem);
    faces_count = faces.size() / 3;
}

void upload_edges(std::vector<float> edges) {
    allocBuffer(edges.size() * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            edge_buffer.buf, edge_buffer.mem
        );
    void* ptr = nullptr;
    VK_CHECK(vkMapMemory(device, edge_buffer.mem, 0, edges.size() * sizeof(float), 0, &ptr));
    std::memcpy(ptr, edges.data(), edges.size() * sizeof(float));
    vkUnmapMemory(device, edge_buffer.mem);
    edges_count = edges.size() / 2;
}

cv::Mat draw(float cam_x, float cam_y, float cam_z, float rot_x, float rot_y, float rot_z) {
    std::chrono::steady_clock::time_point function_enter = std::chrono::steady_clock::now();

    const VkDeviceSize kZeroOffset = 0;

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &beginInfo));

    // Начало render pass: очистка в чёрный / depth = 1.0
    VkClearValue clears[2];
    clears[0].color        = {{0.f, 0.f, 0.f, 1.f}};
    clears[1].depthStencil = {1.f, 0};
    // clears[1].depthStencil = {0.f, 0};

    VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass       = renderPass;
    rpBegin.framebuffer      = framebuffer;
    rpBegin.renderArea       = {{0, 0},
                                {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)}};
    rpBegin.clearValueCount  = 2;
    rpBegin.pClearValues     = clears;
    vkCmdBeginRenderPass(command_buffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    glm::mat4 mats[2];
    glm::vec3 camPos = { cam_x, cam_y, cam_z };

    mats[0] = glm::perspectiveZO(glm::radians(50.f), float(height) / float(width), 0.1f, 200.0f);
    mats[0] = glm::rotate(mats[0], glm::radians(rot_x), glm::vec3{1,0,0});
    mats[0] = glm::rotate(mats[0], glm::radians(rot_y), glm::vec3{0,1,0});
    mats[0] = glm::rotate(mats[0], glm::radians(rot_z), glm::vec3{0,0,1});
    // mats[1] = glm::lookAtLH(
    //     glm::vec3{1, 1, -3},
    //     glm::vec3{0, 0, 0},
    //     glm::vec3{0, 1, 0}
    // );
    mats[1] = glm::translate(glm::mat4(1.f), camPos);
    
    // mats[1] = glm::lookAtRH(
    //     glm::vec3{0, 0, -3},
    //     glm::vec3{0, 0, 0},
    //     glm::vec3{0, -1, 0}
    // );
    // mats[1][1][1] *= -1;
    // mats[1][2][3] *= -1;
    // mats[1][3][3] *= -1;

    std::chrono::steady_clock::time_point preparation_complete = std::chrono::steady_clock::now();

    {
        const float black[4] = {0.0f, 0.0f, 0.0f, 1.f};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, facePipeline);
        vkCmdPushConstants(command_buffer, pipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(black), black);
        vkCmdPushConstants(command_buffer, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           16, sizeof(mats), mats);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &face_buffer.buf, &kZeroOffset);
        // vkCmdDraw(command_buffer, static_cast<uint32_t>(geo.faceVerts.size()), 1, 0, 0);
        vkCmdDraw(command_buffer, faces_count, 1, 0, 0);
    }

    // ── Шаг B: рёбра (белым, с тестом глубины) ───────────────────────────────
    {
        const float white[4] = {1.f, 1.f, 1.f, 1.f};
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline);
        vkCmdSetLineWidth(command_buffer, lineWidth);  // dynamic state — применяется каждый кадр
        vkCmdPushConstants(command_buffer, pipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(white), white);
        vkCmdPushConstants(command_buffer, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           16, sizeof(mats), mats);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &edge_buffer.buf, &kZeroOffset);
        // vkCmdDraw(command_buffer, static_cast<uint32_t>(geo.edgeVerts.size()), 1, 0, 0);
        vkCmdDraw(command_buffer, edges_count, 1, 0, 0);
    }

    vkCmdEndRenderPass(command_buffer);

    VkBufferImageCopy copy{};
    copy.bufferOffset      = 0;
    copy.bufferRowLength   = 0;  // плотная упаковка
    copy.bufferImageHeight = 0;
    copy.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset       = {0, 0, 0};
    copy.imageExtent       = {static_cast<uint32_t>(width),
                              static_cast<uint32_t>(height), 1};
    vkCmdCopyImageToBuffer(command_buffer, colorAttachment.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readBuf, 1, &copy);

    VK_CHECK(vkEndCommandBuffer(command_buffer));

    std::chrono::steady_clock::time_point dispatch = std::chrono::steady_clock::now();

    // ── 4. Отправка и ожидание ────────────────────────────────────────────────
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &command_buffer;
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device, 1, &fence));

    std::chrono::steady_clock::time_point fence = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point copy_img = std::chrono::steady_clock::now();

    auto preparation_complete_mills = std::chrono::duration_cast<std::chrono::milliseconds>(preparation_complete - function_enter).count();
    auto dispatch_mills = std::chrono::duration_cast<std::chrono::milliseconds>(dispatch - preparation_complete).count();
    auto fence_mills = std::chrono::duration_cast<std::chrono::milliseconds>(fence - dispatch).count();
    auto copy_img_mills = std::chrono::duration_cast<std::chrono::milliseconds>(copy_img - fence).count();

    return cv::Mat(int(height), int(width), CV_8UC4, (void*)readPtr);
}

void set_resolution(unsigned int width_, unsigned int height_) {
    width = width_;
    height = height_;
}

// Задать толщину рёбер (применяется с следующего вызова draw()).
// Значение автоматически клэмпится по лимитам GPU.
// Для значений > 1.0 требуется поддержка wideLines (проверяется при init()).
void set_line_width(float w) {
    VkPhysicalDeviceLimits limits{};
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        limits = props.limits;
    }
    // Округление до ближайшего допустимого шага
    const float gran = limits.lineWidthGranularity;
    if (gran > 0.0f)
        w = std::round(w / gran) * gran;

    lineWidth = std::clamp(w,
                           limits.lineWidthRange[0],
                           limits.lineWidthRange[1]);

    if (lineWidth != w)
        fprintf(stderr, "[VK] set_line_width: %.2f → клэмп/округление до %.2f\n", w, lineWidth);
}

}
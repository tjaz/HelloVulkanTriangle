#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>

double currentTime = 0;
double lastTime = glfwGetTime();
int numFrames = 0;
float frameTime = 0.0f;

void calculateFrameRate(GLFWwindow* window) {
	currentTime = glfwGetTime();
	double delta = currentTime - lastTime;

	if (delta >= 1) {
		int framerate(std::max(1, int(numFrames / delta)));
		std::stringstream title;
		title << "Running at " << framerate << " fps";
		glfwSetWindowTitle(window, title.str().c_str());
		lastTime = currentTime;
		numFrames = -1;
		frameTime = float(1000.0 / framerate);
	}

	numFrames++;
}




std::vector<char> readFile(std::string filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to load " + filename);
	}

	size_t filesize{ static_cast<size_t>(file.tellg()) };

	std::vector<char> buffer(filesize);
	file.seekg(0);
	file.read(buffer.data(), filesize);

	file.close();
	return buffer;
}




vk::ShaderModule createShaderModule(std::string filename, vk::Device device) {
	std::vector<char> sourceCode = readFile(filename);
	vk::ShaderModuleCreateInfo moduleInfo;
	moduleInfo.flags = vk::ShaderModuleCreateFlags();
	moduleInfo.codeSize = sourceCode.size();
	moduleInfo.pCode = reinterpret_cast<const uint32_t*>(sourceCode.data());

	try {
		return device.createShaderModule(moduleInfo);
	} catch(const std::exception& e) {
		throw std::runtime_error("Failed to create shader module: " + filename + e.what());
	}
}






int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

	int major, minor, rev;
	glfwGetVersion(&major, &minor, &rev);
	std::cout << "GLFW Version: " << major << "." << minor << "." << rev << std::endl;

	// Get required Vulkan instance extensions
    uint32_t glfwExtensionsCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionsCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionsCount);

    std::cout << "Required Vulkan Instance Extensions:" << std::endl << std::endl;
    for (auto extensionName : extensions) {
        std::cout << extensionName << std::endl;
    }
	std::cout << std::endl;

    // Create a GLFW window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan Cube Demo", nullptr, nullptr);

	if (!glfwVulkanSupported()) {
    	std::cout << "GLFW Vulkan not supported" << std::endl;
    	return -1;
	}

    // Initialize Vulkan
    vk::ApplicationInfo appInfo("Vulkan Cube Demo", 1, nullptr, 0, VK_API_VERSION_1_0);

    // Enable the VK_LAYER_KHRONOS_validation layer
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";

    vk::InstanceCreateInfo instanceInfo(
        vk::InstanceCreateFlags(),
        &appInfo,
        1, &validationLayer,
		static_cast<uint32_t>(extensions.size()), extensions.data()
    );

	// Create vulkan instance
	vk::UniqueInstance instance;
	try {
	    instance = vk::createInstanceUnique(instanceInfo);
	} catch (const vk::SystemError& e) {
	    std::cerr << "Vulkan instance creation failed with error: " << e.what() << std::endl;
	    return -1;
	}
	
	std::cout << "Supported Vulkan-compatible devices:" << std::endl;
	std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();
	if (physicalDevices.empty()) {
	    std::cerr << "No Vulkan-compatible devices found" << std::endl;
	    return -1;
	}
	
	// Select the discrete physical device
	vk::PhysicalDevice physicalDevice = physicalDevices.front();
	// Iterate over the physical devices
    for (const auto& device : physicalDevices) {
        // Access properties of each physical device
        vk::PhysicalDeviceProperties properties = device.getProperties();
        std::cout << "Device Name: " << properties.deviceName << std::endl;
		// Check if the device is discrete or integrated
        std::cout << "Device Type: ";
        switch (properties.deviceType) {
            case vk::PhysicalDeviceType::eIntegratedGpu:
                std::cout << "Integrated GPU" << std::endl;
                break;
            case vk::PhysicalDeviceType::eDiscreteGpu:
                std::cout << "Discrete GPU" << std::endl;
				if (physicalDevice.getProperties().deviceType != properties.deviceType) {
					physicalDevice = device;
				}
                break;
            case vk::PhysicalDeviceType::eVirtualGpu:
                std::cout << "Virtual GPU" << std::endl;
                break;
            case vk::PhysicalDeviceType::eCpu:
                std::cout << "CPU" << std::endl;
                break;
            case vk::PhysicalDeviceType::eOther:
                std::cout << "Other" << std::endl;
                break;
            default:
                std::cout << "Unknown" << std::endl;
                break;
        }
    }
	
	std::cout << std::endl;
	std::cout << "Current physical device: " << physicalDevice.getProperties().deviceName << std::endl;
   

    // Access memory properties
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    // Calculate total memory size in gigabytes
    double totalMemoryGB = static_cast<double>(memoryProperties.memoryHeaps[0].size) / (1 << 30);

    // Print total memory size
    std::cout << "Total Memory: " << totalMemoryGB << " GB" << std::endl;

	
//	// Print supported device extensions
//	auto supportedExtensions = physicalDevice.enumerateDeviceExtensionProperties();
//	std::cout << "Supported Device Extensions:" << std::endl;
//	for (const auto& extension : supportedExtensions) {
//	    std::cout << extension.extensionName << std::endl;
//	}

	// Print supported instance extensions
	auto supportedInstanceExtensions = vk::enumerateInstanceExtensionProperties();
	std::cout << std::endl;
	std::cout << "Supported Instance Extensions:" << std::endl;
	for (const auto& extension : supportedInstanceExtensions) {
	    std::cout << extension.extensionName << std::endl;
	}
	
	// Get queue family properties
	std::cout << std::endl;
	std::cout << "Supported queue families on current physical device:" << std::endl;
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	
    // Find a queue family that supports graphics operations
    uint32_t graphicsQueueFamilyIndex = static_cast<uint32_t>(-1);
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); ++i) {
		const auto& queueFamily = queueFamilyProperties[i];

        // Print information about the queue family
        std::cout << "Queue Family Index: " << i << std::endl;
        std::cout << "Queue Count: " << queueFamily.queueCount << std::endl;

        // Print supported queue flags
        std::cout << "Supported Queue Flags: " << std::endl;
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            std::cout << "Graphics " << std::endl;
			if (graphicsQueueFamilyIndex == static_cast<uint32_t>(-1)) {
				graphicsQueueFamilyIndex = i;
			}
        }
        if (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) {
            std::cout << "Compute " << std::endl;
        }
        if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) {
            std::cout << "Transfer " << std::endl;
        }
        if (queueFamily.queueFlags & vk::QueueFlagBits::eSparseBinding) {
            std::cout << "SparseBinding " << std::endl;
        }
        std::cout << std::endl;
    }

    if (graphicsQueueFamilyIndex == static_cast<uint32_t>(-1)) {
        std::cerr << "No queue family supports graphics operations" << std::endl;
        return -1;
    } else {
		std::cout << "Current graphics queue family index: " << graphicsQueueFamilyIndex << std::endl;
	}

	std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

    // Specify the graphics queue information
    float queuePriority = 1.0f; // Specify the queue priority
    vk::DeviceQueueCreateInfo queueCreateInfo({}, graphicsQueueFamilyIndex, 1, &queuePriority);

    // Specify device features (you might want to customize this based on your requirements)
    vk::PhysicalDeviceFeatures deviceFeatures;

    // Create the device
    vk::DeviceCreateInfo deviceCreateInfo({}, 1, &queueCreateInfo, 0, nullptr, deviceExtensions.size(), deviceExtensions.data(), &deviceFeatures);
	vk::Device device = physicalDevice.createDevice(deviceCreateInfo);
	
    // Create a Vulkan surface
    VkSurfaceKHR surface;
	auto result = glfwCreateWindowSurface(static_cast<VkInstance>(*instance), window, nullptr, &surface);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create window surface" << std::endl;
        return -1;
    }

    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	vk::Extent2D extent = surfaceCapabilities.currentExtent;

	std::cout << "Supported swapchain images:" << std::endl;

	// Get the minimum number of images supported by the swap chain
	uint32_t minImageCount = surfaceCapabilities.minImageCount;
	std::cout << "Minimum number of swap chain images: " << minImageCount << std::endl;
	
	// Get the maximum number of images supported by the swap chain
	uint32_t maxImageCount = surfaceCapabilities.maxImageCount;
	std::cout << "Maximum number of swap chain images: " << maxImageCount << std::endl;

    vk::SurfaceKHR vkSurface(surface);

	// Print supported surface formats
	auto surfaceFormats = physicalDevice.getSurfaceFormatsKHR(vkSurface);
	
	std::cout << std::endl;
	std::cout << "Supported Surface Formats:" << std::endl;
	for (const auto& format : surfaceFormats) {
	    std::cout << "Format: " << vk::to_string(format.format) << ", Color Space: " << vk::to_string(format.colorSpace) << std::endl;
	}

	// Print supported present modes
	auto presentModes = physicalDevice.getSurfacePresentModesKHR(vkSurface);
	std::cout << std::endl;
	std::cout << "Supported Present Modes:" << std::endl;
	for (const auto& mode : presentModes) {
	    std::cout << vk::to_string(mode) << std::endl;
	}
	
	vk::Format swapchainImageFormat = vk::Format::eB8G8R8A8Unorm; 

	// Create a Vulkan swap chain (for simplicity, not handling details like format, present mode, etc.)
    vk::SwapchainCreateInfoKHR createInfo;
    createInfo.surface = vkSurface;
    createInfo.minImageCount = minImageCount;
    createInfo.imageFormat = swapchainImageFormat;
    createInfo.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;

    vk::SwapchainKHR swapChain;
	
	try {
		swapChain = device.createSwapchainKHR(createInfo);

	} catch (vk::SystemError error) {
		std::cerr << "Failed to create swapchain: " << *error.what() << std::endl;
		return -1;
	}

	// Get the swap chain images
	std::vector<vk::Image> swapChainImages = device.getSwapchainImagesKHR(swapChain);

	std::cout << std::endl;
	std::cout << "Number of images in swapchain: " << swapChainImages.size() << std::endl;

	// Create swapchin image views
	std::vector<vk::ImageView> swapChainImageViews;
	for (int i = 0; i < swapChainImages.size(); i++) {
		vk::Image image = swapChainImages[i];
		
		// Specify Image View Create Info
		vk::ImageViewCreateInfo imageViewCreateInfo;
		imageViewCreateInfo.image = image;
		imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
		imageViewCreateInfo.format = swapchainImageFormat;
		imageViewCreateInfo.components = vk::ComponentMapping();
		imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imageViewCreateInfo.subresourceRange.layerCount = 1;

		// Create Image View
		vk::ImageView imageView;
		try {
		    imageView = device.createImageView(imageViewCreateInfo);
			swapChainImageViews.push_back(imageView);

		} catch (const vk::SystemError& e) {
		    std::cerr << "Failed to create image view: " << e.what() << std::endl;
		    return -1;
		}
	}

    // Create a queue
    vk::Queue graphicsQueue = device.getQueue(graphicsQueueFamilyIndex, 0);	
	std::cout << std::endl;

	std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;

	//Vertex input
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	
	// Input assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
	inputAssemblyInfo.flags = vk::PipelineInputAssemblyStateCreateFlags();
	inputAssemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;

	// Vertext shader
	vk::ShaderModule vertexShader = createShaderModule("TriangleVertex.spv", device);
	vk::PipelineShaderStageCreateInfo vertexShaderInfo;
	vertexShaderInfo.flags = vk::PipelineShaderStageCreateFlags();
	vertexShaderInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertexShaderInfo.module = vertexShader;
	vertexShaderInfo.pName = "main";
	shaderStages.push_back(vertexShaderInfo);

	// Viewport and scissor
	vk::Viewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = extent.width;
	viewport.height = extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vk::Rect2D scissor;
	scissor.offset.x = 0.0f;
	scissor.offset.y = 0.0f;
	scissor.extent = extent;
	vk::PipelineViewportStateCreateInfo viewportStateInfo;
	viewportStateInfo.flags = vk::PipelineViewportStateCreateFlags();
	viewportStateInfo.viewportCount = 1;
	viewportStateInfo.pViewports = &viewport;
	viewportStateInfo.scissorCount = 1;
	viewportStateInfo.pScissors = &scissor;

	// Rasterizer
	vk::PipelineRasterizationStateCreateInfo rasterizerInfo;
	rasterizerInfo.flags = vk::PipelineRasterizationStateCreateFlags();
	rasterizerInfo.depthClampEnable = VK_FALSE;
	rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizerInfo.polygonMode = vk::PolygonMode::eFill;
	rasterizerInfo.lineWidth = 1.0;
	rasterizerInfo.cullMode = vk::CullModeFlagBits::eBack;
	rasterizerInfo.frontFace = vk::FrontFace::eClockwise;
	rasterizerInfo.depthBiasEnable = VK_FALSE;

	// Fragment shader
	vk::ShaderModule fragmentShader = createShaderModule("TriangleFragment.spv", device);
	vk::PipelineShaderStageCreateInfo fragmentShaderInfo;
	fragmentShaderInfo.flags = vk::PipelineShaderStageCreateFlags();
	fragmentShaderInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragmentShaderInfo.module = fragmentShader;
	fragmentShaderInfo.pName = "main";
	shaderStages.push_back(fragmentShaderInfo);

	// Multisampling
	vk::PipelineMultisampleStateCreateInfo multisamplingInfo;
	multisamplingInfo.flags = vk::PipelineMultisampleStateCreateFlags();
	multisamplingInfo.sampleShadingEnable = VK_FALSE;
	multisamplingInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;

	// Color blend
	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
 	colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	colorBlendAttachment.blendEnable = VK_FALSE;
	vk::PipelineColorBlendStateCreateInfo colorBlendingInfo;
	colorBlendingInfo.flags = vk::PipelineColorBlendStateCreateFlags();
	colorBlendingInfo.logicOpEnable = VK_FALSE;
	colorBlendingInfo.logicOp = vk::LogicOp::eCopy;
	colorBlendingInfo.attachmentCount = 1;
	colorBlendingInfo.pAttachments = &colorBlendAttachment;
	colorBlendingInfo.blendConstants[0] = 0.0f;
	colorBlendingInfo.blendConstants[1] = 0.0f;
	colorBlendingInfo.blendConstants[2] = 0.0f;
	colorBlendingInfo.blendConstants[3] = 0.0f;

	// Create pipeline layout
	vk::PipelineLayoutCreateInfo layoutInfo;
	layoutInfo.flags = vk::PipelineLayoutCreateFlags();
	layoutInfo.setLayoutCount = 0;
	layoutInfo.pushConstantRangeCount = 0;
	vk::PipelineLayout pipelineLayout;
	try {
		pipelineLayout = device.createPipelineLayout(layoutInfo);
	} catch (vk::SystemError error) {
		std::cerr << "Failed to create pipeline layout: " << *error.what() << std::endl;
		return -1;
	}

	// Create renderpass
	vk::AttachmentDescription colorAttachment;
	colorAttachment.flags = vk::AttachmentDescriptionFlags();
	colorAttachment.format = swapchainImageFormat;
	colorAttachment.samples = vk::SampleCountFlagBits::e1;
	colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
	colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
	colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
	colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
	
	vk::AttachmentReference colorAttachmentRef;
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::SubpassDescription subpass;
	subpass.flags = vk::SubpassDescriptionFlags();
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	vk::RenderPassCreateInfo renderpassInfo;
	renderpassInfo.flags = vk::RenderPassCreateFlags();
	renderpassInfo.attachmentCount = 1;
	renderpassInfo.pAttachments = &colorAttachment;
	renderpassInfo.subpassCount = 1;
	renderpassInfo.pSubpasses = &subpass;

	vk::RenderPass renderPass;
	try {
		renderPass = device.createRenderPass(renderpassInfo);
	} catch (vk::SystemError error) {
		std::cerr <<"Failed to create the renderpass: " << *error.what() << std::endl;
		return -1;
	}
		
	// Create graphics pipeline
	vk::GraphicsPipelineCreateInfo pipelineInfo;
	pipelineInfo.flags = vk::PipelineCreateFlags();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
	pipelineInfo.pViewportState = &viewportStateInfo;
	pipelineInfo.pRasterizationState = &rasterizerInfo;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pMultisampleState = &multisamplingInfo;
	pipelineInfo.pColorBlendState = &colorBlendingInfo;	
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.basePipelineHandle = nullptr;

	vk::Pipeline graphicsPipeline;
	try {
		graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
	} catch(vk::SystemError error) {
		std::cerr << "Failed to create graphics pipeline: " << *error.what() << std::endl;
		return -1;
	}

	// Create framebuffers
	std::vector<vk::Framebuffer> framebuffers;
	framebuffers.resize(swapChainImageViews.size());
	for (int i = 0; i < framebuffers.size(); i++) {
		vk::FramebufferCreateInfo framebufferInfo;
		framebufferInfo.flags = vk::FramebufferCreateFlags();
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = &swapChainImageViews[i];
		framebufferInfo.width = extent.width;
		framebufferInfo.height = extent.height;
		framebufferInfo.layers = 1;

		try {
			framebuffers[i] = device.createFramebuffer(framebufferInfo);
			std::cout << "Created framebuffer for frame: " << i << std::endl;
		} catch (vk::SystemError error) {
			std::cerr << "Failed to create framebuffer for frame: " << i << " and reason: " << *error.what() << std::endl;
			return -1;
		}
	}

	// Create command pool
	vk::CommandPoolCreateInfo poolInfo;
	poolInfo.flags = vk::CommandPoolCreateFlags() | vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;

	vk::CommandPool commandPool;
	try {
		commandPool = device.createCommandPool(poolInfo);
	} catch(vk::SystemError error) {
		std::cerr << "Failed to create command pool: " << *error.what() << std::endl;
		return -1;
	}

	// Allocate command buffers
	vk::CommandBufferAllocateInfo allocateInfo;
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = vk::CommandBufferLevel::ePrimary;
	allocateInfo.commandBufferCount = 1;

	std::vector<vk::CommandBuffer> commandBuffers;
	commandBuffers.resize(framebuffers.size());
	for (int i = 0; i < commandBuffers.size(); i++) {
		try {
			commandBuffers[i] = device.allocateCommandBuffers(allocateInfo)[0];
			std::cout << "Allocated command buffer for frame: " << i << std::endl;
		} catch(vk::SystemError error) {
			std::cerr << "Failed to allocate command buffer for frame: " << i << std::endl;
		}
	}

	// vk::CommandBuffer mainCommandBuffer;
	// try {
	// 	mainCommandBuffer = device.allocateCommandBuffers(allocateInfo)[0];
	// 	std::cout << "Allocated the main command buffer" << std::endl;
	// } catch(vk::SystemError error) {
	// 	std::cerr << "Failed to create main command buffer: " << *error.what() << std::endl;
	// 	return -1;
	// }

	// Destroy shaders
	device.destroyShaderModule(vertexShader);
	device.destroyShaderModule(fragmentShader);

	// Create sempahores
	std::cout << std::endl;
	vk::SemaphoreCreateInfo semaphoreInfo;
	semaphoreInfo.flags = vk::SemaphoreCreateFlags();
	
	std::array<vk::Semaphore, 2> semaphores;
	for (auto& semaphore : semaphores) {
		try {
			semaphore = device.createSemaphore(semaphoreInfo);
		} catch (vk::SystemError error) {
			std::cerr << "Failed to create the semaphore: " << *error.what() << std::endl;
			return -1;
		}
	}
	std::cout << "Successfully created sempahores: " << semaphores.size() << std::endl;

	// Create fence
	vk::FenceCreateInfo fenceInfo;
	fenceInfo.flags = vk::FenceCreateFlags() | vk::FenceCreateFlagBits::eSignaled;

	vk::Fence fence;
	try {
		fence = device.createFence(fenceInfo);
		std::cout << "Successfully created fence" << std::endl;
	} catch (vk::SystemError error) {
		std::cerr << "Failed to create fence: " << *error.what() << std::endl;
		return -1;
	}

	std::cout << std::endl;

	// Main loop
	while (!glfwWindowShouldClose(window)) {
	    glfwPollEvents();
		
		if (device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
			std::cerr << "Failed to wait on the fence" << std::endl;
			return -1;
		}
		if (device.resetFences(1, &fence) != vk::Result::eSuccess) {
			std::cerr << "Failed to reset the fence" << std::endl;
			return -1;
		}

		uint32_t imageIndex = device.acquireNextImageKHR(swapChain, UINT64_MAX, semaphores[0], nullptr).value;
		
	    // Rendering code goes here
		vk::ClearValue clearColor(std::array<float, 4>{0.0, 0.0, 0.0, 1.0});

		// Create renderPass info
		vk::RenderPassBeginInfo renderPassInfo;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = framebuffers[imageIndex];
		renderPassInfo.renderArea.offset.x = 0;
		renderPassInfo.renderArea.offset.y = 0;
		renderPassInfo.renderArea.extent = extent;
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;
		
		// Select the comand buffer
		auto& commandBuffer = commandBuffers[imageIndex];
		commandBuffer.reset();
		vk::CommandBufferBeginInfo beginInfo;
		try {
			commandBuffer.begin(beginInfo);
		} catch (vk::SystemError error) {
			std::cerr << "Failed to begin rendering command buffer: " << *error.what() << std::endl;
			return -1;
		}
		commandBuffer.beginRenderPass(&renderPassInfo, vk::SubpassContents::eInline);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRenderPass();
		try {
			commandBuffer.end();
		} catch (vk::SystemError error) {
			std::cerr << "Failed to finish recording command buffer: " << *error.what() << std::endl;
			return -1;
		}

	    // End rendering

		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		
		// Configure submitInfo
		vk::SubmitInfo submitInfo;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &semaphores[0];
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
	    submitInfo.signalSemaphoreCount = 1;
	    submitInfo.pSignalSemaphores = &semaphores[1];

		try {
	    	graphicsQueue.submit(submitInfo, fence);
		} catch (vk::SystemError error) {
			std::cerr << "Failed to submid draw command buffer: " << std::endl;
			return -1;
		}

	    // Present the image using the swap chain's queue

        // Configure presentInfo
        vk::PresentInfoKHR presentInfo;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &semaphores[1];

        // Submit the present request
        if (graphicsQueue.presentKHR(presentInfo) != vk::Result::eSuccess) {
			std::cerr << "Failed to present the draw image on queue" << std::endl;
			return -1;
		}

		calculateFrameRate(window);
	}

	device.waitIdle();

    // Cleanup
	for (auto& semaphore : semaphores) {
		device.destroySemaphore(semaphore);
	}
	for (auto& imageView : swapChainImageViews) {
		device.destroyImageView(imageView);
	}
	for (auto& framebuffer : framebuffers) {
		device.destroyFramebuffer(framebuffer);	
	}
	device.destroyFence(fence);
	device.destroyCommandPool(commandPool);
	device.destroyPipeline(graphicsPipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyRenderPass(renderPass);
    device.destroySwapchainKHR(swapChain);
    instance->destroySurfaceKHR(vkSurface);
    device.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}


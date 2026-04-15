/*
* V2D test for Spacemit
* Copyright (C) 2023 Spacemit Co., Ltd.
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include "pthread.h"
#include "/home/perise/source/k1x-v2d-main/include/v2d_type.h"
#include "/home/perise/source/k1x-v2d-main/include/v2d_api.h"
#ifndef O_CLOEXEC
#  define O_CLOEXEC 02000000
#endif
#include <poll.h>
#include <unistd.h>

int gFd = -1;

typedef enum SPACEMIT_V2D_TASK_TYPE_E
{
	TASK_NONE   = 0,
	ROTATION   = 1,
	CSC  = 2,
	FILL = 3,
	BLEND = 3,
} V2D_TASK_TYPE_E;

typedef struct SPACEMIT_V2D_TASK_S
{
	V2D_SUBMIT_TASK_S stV2dTask;
	struct SPACEMIT_V2D_TASK_S *pNext;
} V2D_TASK_S;

typedef struct SPACEMIT_VGS_JOB_S
{
	uint32_t count;
	V2D_TASK_TYPE_E  currentTaskState;
	V2D_TASK_S *pHead;
	V2D_TASK_S *pTail;
} V2D_JOB_S;

#define DEV_NAME "/dev/v2d_dev"
#define MAX_TASK_LIST_LENGTH 64

static void freeList(V2D_TASK_S * pHead)
{
	V2D_TASK_S * p;
	while(pHead != NULL)
	{
		p = pHead;
		pHead = pHead->pNext;
		free(p);
		p = NULL;
	}
}

static int sync_wait(int fd, int timeout)
{
	struct pollfd fds;
	int ret = 0;
	if (fd < 0) {
		return -1;
	}
	fds.fd = fd;
	fds.events = POLLIN;
	do {
		ret = poll(&fds, 1, timeout);
		if (ret > 0) {
			if (fds.revents & (POLLERR | POLLNVAL)) {
				return -1;
			}
			return 0;
		} else if (ret == 0) {
			return -1;
		}
	} while (ret == -1 );
	return ret;
}

static int v2d_lock_async(int fence_fd)
{
	int ret = 0;
	if (fence_fd >= 0)
	{
		ret = sync_wait(fence_fd, 3000);
		close(fence_fd);
	}
	return ret;
}

static int V2dSubmitJob(V2D_JOB_S *pstV2dJob)
{
	int ret;
	int i;
	V2D_TASK_S * curNode = pstV2dJob->pHead;
	for (i=0; i<pstV2dJob->count; i++)
	{
		curNode->stV2dTask.acquireFencefd = -1;
		curNode->stV2dTask.completeFencefd = -1;
		ret = write(gFd, curNode, sizeof(V2D_TASK_S));
		if (ret != sizeof(V2D_TASK_S))
		{
			printf("Failed to submit V2D task!\n");
			return FAILURE;
		}
		curNode = curNode->pNext;
	}

	curNode = pstV2dJob->pHead;
	for (i=0; i<pstV2dJob->count; i++)
	{
		ret = v2d_lock_async(curNode->stV2dTask.completeFencefd);
		if(curNode->stV2dTask.acquireFencefd > 0)
			close(curNode->stV2dTask.acquireFencefd);
		curNode = curNode->pNext;
	}
	return ret;
}

int32_t V2D_BeginJob(V2D_HANDLE *pHandle)
{
	V2D_JOB_S *pstV2dJob = NULL;
	pstV2dJob = (V2D_JOB_S*)malloc(sizeof(V2D_JOB_S));
	if (!pstV2dJob)
	{
		printf("Failed to malloc v2d job\n");
		return FAILURE;
	}
	memset(pstV2dJob,0,sizeof(V2D_JOB_S));
	pstV2dJob->currentTaskState = TASK_NONE;
	pstV2dJob->pHead = NULL;
	pstV2dJob->pTail = NULL;
	*pHandle = (uint64_t)(pstV2dJob);
	return SUCCESS;
}

int32_t V2D_EndJob(V2D_HANDLE hHandle)
{
	int ret = 0;
	if(hHandle==0)
		return FAILURE;
	V2D_JOB_S *pstV2dJob=(V2D_JOB_S *)hHandle;

	if(gFd < 0)
	{
		gFd = open(DEV_NAME, O_RDWR|O_CLOEXEC|O_NONBLOCK);
		if(gFd >= 0)
		{
			ret = V2dSubmitJob(pstV2dJob);
		}else{
			gFd = -1;
			printf("Failed to open device file %s\n", DEV_NAME);
			ret = FAILURE;
		}
	}else{
		ret = V2dSubmitJob(pstV2dJob);
	}
	freeList(pstV2dJob->pHead);
	pstV2dJob->pTail = NULL;
	if(pstV2dJob)
	{
		free(pstV2dJob);
		pstV2dJob = NULL;
	}
	hHandle=-1;
	return ret;
}
int32_t V2D_AddFillTask(V2D_HANDLE hHandle, V2D_SURFACE_S *pstDst, V2D_AREA_S *pstDstRect,  V2D_FILLCOLOR_S *pstFillColor)
{
	V2D_PARAM_S *pstParam;
	V2D_SURFACE_S *pstSrc, *pstOut;
	V2D_BLEND_CONF_S *pstBlendConf;
	V2D_BLEND_LAYER_CONF_S *pstBlendLayerConf;

	if (hHandle==0)
		return FAILURE;

	V2D_JOB_S *pstV2dJob=(V2D_JOB_S *)hHandle;
	pstV2dJob->count++;
	if (pstV2dJob->count>MAX_TASK_LIST_LENGTH)
	{
		printf("Faied to add fill task, the task list length exceeds %d\n", MAX_TASK_LIST_LENGTH);
		pstV2dJob->count--;
		return FAILURE;
	}
	pstV2dJob->currentTaskState = FILL;
	V2D_TASK_S * pNew = (V2D_TASK_S *)malloc(sizeof(V2D_TASK_S));
	if(pNew == NULL)
	{
		printf("The new node assignment failed and the program terminated!\n");
		exit(-1);
	}
	memset(pNew,0,sizeof(V2D_TASK_S));
	pstParam = &pNew->stV2dTask.param;
	//config layer0 input solid color
	pstParam->l0_csc = V2D_CSC_MODE_BUTT;
	pstSrc = &pstParam->layer0;
	pstSrc->solidcolor.enable = 1;
	memcpy(&pstSrc->solidcolor.fillcolor, pstFillColor, sizeof(V2D_FILLCOLOR_S));
	//config output
	if (pstDst) {
		memcpy(&pstParam->dst, pstDst, sizeof(V2D_SURFACE_S));
	}
	if (pstDstRect) {
		memcpy(&pstParam->dst_rect, pstDstRect, sizeof(V2D_AREA_S));
	}
	//config blend
	pstBlendConf = &pstParam->blendconf;
	pstBlendConf->blend_cmd = V2D_BLENDCMD_ALPHA;
	pstBlendConf->bgcolor.enable = 0;
	pstBlendLayerConf = &pstBlendConf->blendlayer[0];
	pstBlendLayerConf->blend_area.x = pstDstRect->x;
	pstBlendLayerConf->blend_area.y = pstDstRect->y;
	pstBlendLayerConf->blend_area.w = pstDstRect->w;
	pstBlendLayerConf->blend_area.h = pstDstRect->h;

	pNew->pNext=NULL;
	if (!pstV2dJob->pHead) {
		pstV2dJob->pHead = pNew;
	} else {
		pstV2dJob->pTail->pNext = pNew;
	}
	pstV2dJob->pTail = pNew;

	return SUCCESS;
}

int32_t V2D_AddBitblitTask(V2D_HANDLE hHandle, V2D_SURFACE_S *pstDst, V2D_AREA_S *pstDstRect, V2D_SURFACE_S *pstSrc,
                                V2D_AREA_S *pstSrcRect, V2D_CSC_MODE_E enCSCMode)
{
	V2D_PARAM_S *pstParam;
	V2D_SURFACE_S *pstInput, *pstOutput;
	V2D_BLEND_CONF_S *pstBlendConf;
	V2D_BLEND_LAYER_CONF_S *pstBlendLayerConf;
	if (hHandle==0)
		return FAILURE;
	V2D_JOB_S *pstV2dJob=(V2D_JOB_S *)hHandle;
	pstV2dJob->count++;
	if (pstV2dJob->count>MAX_TASK_LIST_LENGTH)
	{
		printf("Faied to add bitblit task, the task list length exceeds %d\n", MAX_TASK_LIST_LENGTH);
		pstV2dJob->count--;
		return FAILURE;
	}
	pstV2dJob->currentTaskState = TASK_NONE;
	V2D_TASK_S * pNew = (V2D_TASK_S *)malloc(sizeof(V2D_TASK_S));
	if(pNew == NULL)
	{
		printf("The new node assignment failed and the program terminated!\n");
		exit(-1);
	}
	memset(pNew,0,sizeof(V2D_TASK_S));
	pstParam = &pNew->stV2dTask.param;
	//config input
	pstParam->l0_csc = V2D_CSC_MODE_BUTT;
	memcpy(&pstParam->layer0, pstSrc, sizeof(V2D_SURFACE_S));
	memcpy(&pstParam->l0_rect, pstSrcRect, sizeof(V2D_AREA_S));
	//config output
	memcpy(&pstParam->dst, pstDst, sizeof(V2D_SURFACE_S));
	memcpy(&pstParam->dst_rect, pstDstRect, sizeof(V2D_AREA_S));
	//config blend
	pstBlendConf = &pstParam->blendconf;
	pstBlendConf->bgcolor.enable = 0;
	pstBlendLayerConf = &pstBlendConf->blendlayer[0];
	pstBlendLayerConf->blend_area.x = pstDstRect->x;
	pstBlendLayerConf->blend_area.y = pstDstRect->y;
	pstBlendLayerConf->blend_area.w = pstDstRect->w;
	pstBlendLayerConf->blend_area.h = pstDstRect->h;

	pNew->pNext=NULL;
	if (!pstV2dJob->pHead) {
		pstV2dJob->pHead = pNew;
	} else {
		pstV2dJob->pTail->pNext = pNew;
	}
	pstV2dJob->pTail = pNew;

	return SUCCESS;

}
int32_t V2D_AddBlendTask(V2D_HANDLE hHandle, 
                             V2D_SURFACE_S *pstBackGround,
                             V2D_AREA_S *pstBackGroundRect,
                             V2D_SURFACE_S *pstForeGround,
                             V2D_AREA_S *pstForeGroundRect,
                             V2D_SURFACE_S *pstMask,
                             V2D_AREA_S *pstMaskRect,
                             V2D_SURFACE_S *pstDst,
                             V2D_AREA_S *pstDstRect,
                             V2D_BLEND_CONF_S *pstBlendConf,
                             V2D_ROTATE_ANGLE_E enForeRotateAngle,
                             V2D_ROTATE_ANGLE_E enBackRotateAngle,
                             V2D_CSC_MODE_E enForeCSCMode,
                             V2D_CSC_MODE_E enBackCSCMode,
                             V2D_PALETTE_S *pstPalette,
                             V2D_DITHER_E dither)
{
	V2D_PARAM_S *pstParam;

	if (hHandle==0)
		return FAILURE;
	V2D_JOB_S *pstV2dJob=(V2D_JOB_S *)hHandle;
	pstV2dJob->count++;
	if (pstV2dJob->count>MAX_TASK_LIST_LENGTH)
	{
		printf("Faied to add blend task, the task list length exceeds %d\n", MAX_TASK_LIST_LENGTH);
		pstV2dJob->count--;
		return FAILURE;
	}
	pstV2dJob->currentTaskState = BLEND;
	V2D_TASK_S * pNew = (V2D_TASK_S *)malloc(sizeof(V2D_TASK_S));
	if(pNew == NULL)
	{
		printf("The new node assignment failed and the program terminated!\n");
		exit(-1);
	}
	memset(pNew,0,sizeof(V2D_TASK_S));
	pstParam = &pNew->stV2dTask.param;
	pstParam->l0_csc = enBackCSCMode;
	pstParam->l1_csc = enForeCSCMode;
	pstParam->l0_rt  = enBackRotateAngle;
	pstParam->l1_rt  = enForeRotateAngle;
	pstParam->dither = dither;
	if (pstBackGround) {
		memcpy(&pstParam->layer0, pstBackGround, sizeof(V2D_SURFACE_S));
	}
	if (pstBackGroundRect) {
		memcpy(&pstParam->l0_rect, pstBackGroundRect, sizeof(V2D_AREA_S));
	}
	if (pstForeGround) {
		memcpy(&pstParam->layer1, pstForeGround, sizeof(V2D_SURFACE_S));
	}
	if (pstForeGroundRect) {
		memcpy(&pstParam->l1_rect, pstForeGroundRect, sizeof(V2D_AREA_S));
	}
	if (pstMask) {
		memcpy(&pstParam->mask, pstMask, sizeof(V2D_SURFACE_S));
	}
	if (pstMaskRect) {
		memcpy(&pstParam->mask_rect, pstMaskRect, sizeof(V2D_AREA_S));
	}
	if (pstDst) {
		memcpy(&pstParam->dst, pstDst, sizeof(V2D_SURFACE_S));
	}
	if (pstDstRect) {
		memcpy(&pstParam->dst_rect, pstDstRect, sizeof(V2D_AREA_S));
	}
	if (pstBlendConf) {
		memcpy(&pstParam->blendconf, pstBlendConf, sizeof(V2D_BLEND_CONF_S));
	}
	if (pstPalette) {
		memcpy(&pstParam->palette, pstPalette, sizeof(V2D_PALETTE_S));
	}
	pNew->pNext=NULL;
	if (!pstV2dJob->pHead) {
		pstV2dJob->pHead = pNew;
	} else {
		pstV2dJob->pTail->pNext = pNew;
	}
	pstV2dJob->pTail = pNew;

	return SUCCESS;
}


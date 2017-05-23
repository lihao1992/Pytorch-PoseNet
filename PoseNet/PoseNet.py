""" PoseNet """
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


# Inception (based on googlenet)
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.c1x1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1, stride=1),
            nn.ReLU(True)
        )

        # 1x1 conv -> 3x3 conv
        self.c3x3 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # 1x1 conv -> 5x5 conv
        self.c5x5 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv
        self.poolconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1, stride=1),
            nn.ReLU(True)
        )

    def forward(self, x):

        y1 = self.c1x1(x)
        y2 = self.c3x3(x)
        y3 = self.c5x5(x)
        y4 = self.poolconv(x)
        output = [y1,y2,y3,y4]

        return torch.cat(output, 1)




# PoseNet
class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )

        self.icp_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.icp_3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.icp_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.cls1_re = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU(True)
        self.cls1_fc_pose = nn.Linear(128, 1024)
        self.relu2 = nn.ReLU(True)
        self.cls1_fc_pose_xyz = nn.Linear(1024, 3)
        self.cls1_fc_pose_wpqr = nn.Linear(1024, 4)

        self.icp_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.icp_4c = Inception(512, 128, 128, 256, 24, 64, 64)

        self.icp_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.avgpool2 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.cls2_re = nn.Conv2d(528, 128, kernel_size=1, stride=1)
        self.relu3 = nn.ReLU(True)
        self.cls2_fc_pose = nn.Linear(128, 1024)
        self.relu4 = nn.ReLU(True)
        self.cls2_fc_pose_xyz = nn.Linear(1024, 3)
        self.cls2_fc_pose_wpqr = nn.Linear(1024, 4)

        self.icp_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.icp_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.icp_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool3 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dp = nn.Dropout2d(p=0.4)
        self.cls3_fc_pose = nn.Linear(1024, 2048)
        self.relu = nn.ReLU(True)

        self.cls3_fc_pose_xyz = nn.Linear(2048, 3)
        self.cls3_fc_pose_wpqr = nn.Linear(2048, 4)

    def forward(self, x):

        out = self.layer1(x)
        out = self.icp_3a(out)
        out = self.icp_3b(out)
        out = self.icp_4a(out)

        out_pose1_avg = self.avgpool1(out)
        out_pose1_re = self.cls1_re(out_pose1_avg)
        out_pose1_re = out_pose1_re.view(out_pose1_re.size(0), -1)
        out_pose1 = self.cls1_fc_pose(out_pose1_re)
        out_pose1_xyz = self.cls1_fc_pose_xyz(out_pose1)
        out_pose1_wpqr = self.cls1_fc_pose_wpqr(out_pose1)

        out = self.icp_4b(out)
        out = self.icp_4c(out)
        out = self.icp_4d(out)

        out_pose2_avg = self.avgpool2(out)
        out_pose2_re = self.cls2_re(out_pose2_avg)
        out_pose2_re = out_pose2_re.view(out_pose2_re.size(0), -1)
        out_pose2 = self.cls2_fc_pose(out_pose2_re)
        out_pose2_xyz = self.cls2_fc_pose_xyz(out_pose2)
        out_pose2_wpqr = self.cls2_fc_pose_wpqr(out_pose2)

        out = self.icp_4e(out)
        out = self.icp_5a(out)
        out = self.icp_5b(out)
        out = self.avgpool3(out)
        out = out.view(out.size(0), -1)
        out = self.cls3_fc_pose(out)
        out_pose3_xyz = self.cls3_fc_pose_xyz(out)
        out_pose3_wpqr = self.cls3_fc_pose_wpqr(out)

        # return torch.cat([out_pose3_xyz, out_pose3_wpqr], 1)

        out_final = [out_pose1_xyz, out_pose1_wpqr, out_pose2_xyz, out_pose2_wpqr, out_pose3_xyz, out_pose3_wpqr]

        return torch.cat(out_final, 1)

net = PoseNet().cuda()

print 1
print 1
print 1
print 1
print 1
print 1

#x = torch.randn(1, 3, 227, 227)
#y = net(Variable(x).cuda())

#print (y.size())







